import re
import sys
import os
import io
import json
import ast
import base64
import csv
import logging
import urllib.request
import urllib.error
from statistics import median
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime, timezone

from janome.tokenizer import Tokenizer
import uuid

import nltk
from nltk import pos_tag
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import numpy as np
from pdf2image import convert_from_bytes
from neo4j import GraphDatabase
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# content_info.py をモジュールとしてインポート
import content_info

# ==========================================
# ログ・環境設定
# ==========================================
logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_JSON_PATH = os.path.join(BASE_DIR, "parent_contentpage.json")
TFIDF_CSV_PATH = os.path.join(BASE_DIR, "unit_tfidf_matrix_normalized.csv")

# グローバル変数の定義
global_infra = None
global_math_estimator = None


# ==========================================
# 1. 共通インフラクラス (DB・API接続)
# ==========================================
class CoreInfra:
    """DB接続やOpenAIクライアントなどの共通リソースを管理するクラス"""
    def __init__(self, neo4j_url, neo4j_auth, openai_api_key):
        self.driver = GraphDatabase.driver(neo4j_url, auth=neo4j_auth)
        self.client = OpenAI(api_key=openai_api_key)

    def close(self):
        self.driver.close()

    def get_embedding(self, text):
        """全教科共通で使用するテキスト埋め込み取得処理"""
        text = text.replace("\n", " ")
        if not text: return np.zeros(1536)
        try:
            response = self.client.embeddings.create(input=[text], model="text-embedding-3-small")
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI Embedding Error: {e}")
            return np.zeros(1536)


# ==========================================
# 2. 数学用 単元推定クラス
# ==========================================
class MathUnitEstimator:
    """数学特有のデータ(TF-IDF, 教科書ベクトル)と単元推定ロジックを管理するクラス"""
    def __init__(self, core_infra, reference_json_path, tfidf_csv_path):
        self.infra = core_infra
        self.driver = core_infra.driver
        
        self._load_reference_data(reference_json_path)
        self._load_tfidf_data(tfidf_csv_path)
        
        logger.info("Loading math concepts from Neo4j...")
        self.concepts = self._fetch_concepts()

    def _load_reference_data(self, path):
        logger.info(f"Loading reference data from {path}...")
        self.reference_records = []
        vsm_list = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            for record in raw_data:
                vsm_val = record.get('vsm')
                vec = None
                if isinstance(vsm_val, list) and vsm_val: 
                    vec = np.array(vsm_val)
                elif isinstance(vsm_val, str) and vsm_val.strip():
                    try:
                        parsed = ast.literal_eval(vsm_val)
                        if parsed: vec = np.array(parsed)
                    except: pass
                
                if vec is not None:
                    self.reference_records.append(record)
                    vsm_list.append(vec)
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            
        self.reference_vsm_matrix = np.vstack(vsm_list) if vsm_list else np.array([])

    def _load_tfidf_data(self, path):
        logger.info(f"Loading TF-IDF matrix from {path}...")
        self.tfidf_data = {}
        self.tfidf_keywords = set()
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    self.tfidf_keywords = set(reader.fieldnames) - {'unit_id'}
                for row in reader:
                    uid = row.get('unit_id')
                    if uid:
                        scores = {k: float(v) for k, v in row.items() if k != 'unit_id' and v and float(v) > 0}
                        self.tfidf_data[str(uid)] = scores
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")

    def _fetch_concepts(self):
        query = "MATCH (c:Concept) WHERE toInteger(c.concept_id) > 989999999 RETURN c.concept_name AS name"
        try:
            with self.driver.session() as session:
                result = session.run(query)
                return sorted([record["name"] for record in result], key=len, reverse=True)
        except Exception as e:
            logger.error(f"Neo4j Error: {e}")
            return []

    def process_math_page(self, text_info, contentssection_id, pre_calculated_vsm):
        """数学用の単元推定・キーワードリンク処理"""
        text_info_tmp = text_info
        extracted_counts = {}
        for concept_name in self.concepts:
            if not text_info_tmp: break
            count = text_info_tmp.count(concept_name)
            if count > 0:
                extracted_counts[concept_name] = count
                text_info_tmp = text_info_tmp.replace(concept_name, "")
                
        if extracted_counts:
            keyword_data = [{"name": k, "count": v} for k, v in extracted_counts.items()]
            query_link_concepts = """
            MATCH (bs:BookSection {contentssection_id: $bs_id})
            UNWIND $keyword_data AS kw
            MATCH (c:Concept {concept_name: kw.name})
            WHERE toInteger(c.concept_id) > 989999999
            
            MERGE (bs)-[r:CONTAINS]->(c)
            SET r.num = kw.count
            """
            try:
                with self.driver.session() as session:
                    session.run(query_link_concepts, bs_id=contentssection_id, keyword_data=keyword_data)
            except Exception as e:
                logger.error(f"Neo4j Concept Link Error: {e}")

        vsm = np.array(pre_calculated_vsm).reshape(1, -1)
        valid_keywords = [kw for kw in extracted_counts.keys() if kw in self.tfidf_keywords]
        candidate_parent_unit_ids = set()
        
        if valid_keywords:
            for uid, scores_map in self.tfidf_data.items():
                if sum(scores_map.get(kw, 0.0) for kw in valid_keywords) > 0:
                    candidate_parent_unit_ids.add(uid)

        if self.reference_vsm_matrix.size == 0: return []
            
        similarities = cosine_similarity(vsm, self.reference_vsm_matrix)[0]
        grouped_sims = {}

        for i, record in enumerate(self.reference_records):
            uid = str(record.get('unit_id', ''))
            if candidate_parent_unit_ids and uid not in candidate_parent_unit_ids: continue
            sub_id = str(record.get('subunit_id', ''))
            if not uid or not sub_id: continue

            if (uid, sub_id) not in grouped_sims:
                grouped_sims[(uid, sub_id)] = []
            grouped_sims[(uid, sub_id)].append(similarities[i])

        if not grouped_sims: return []

        results = []
        for (parent_id, sub_id), sim_values in grouped_sims.items():
            top_n_values = sorted(sim_values, reverse=True)[:3]
            if top_n_values:
                results.append({"unit_id": parent_id, "subunit_id": sub_id, "score": median(top_n_values)})
            
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)[:3]
        
        query_connect = """
        MATCH (bs:BookSection {contentssection_id: $bs_id})
        MATCH (u:Unit {unit_id: $target_unit_id})
        MERGE (bs)-[r:RELATED_TO]->(u)
        SET r.ratio = $score, r.rank =$rank
        """
        try:
            with self.driver.session() as session:
                for i, cand in enumerate(sorted_results, 1):
                    if cand['subunit_id']:
                        session.run(query_connect, bs_id=contentssection_id, target_unit_id=cand['subunit_id'], score=float(cand['score']), rank=i)
        except Exception as e:
            logger.error(f"Neo4j Connect Error: {e}")
            
        return sorted_results


# ==========================================
# 3. 教科別プロセッサ
# ==========================================
class BaseSubjectProcessor:
    TEXT_PROMPT = "あなたの仕事は画像に含まれている文章を抽出してテキストで出力することです。"
    IMAGE_PROMPT = "あなたの仕事は画像に含まれている図の説明をテキストで出力することです。"

    # object_id を引数に追加
    def __init__(self, contents_id, object_id, core_infra):
        self.contents_id = contents_id
        self.object_id = object_id
        self.infra = core_infra
        self.client = core_infra.client
        self.driver = core_infra.driver

    def _encode_image(self, pil_image):
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _call_openai_vision(self, base64_image):
        comp_text = self.client.chat.completions.create(
            model="o4-mini",
            messages=[{"role": "user", "content": [{"type": "text", "text": self.TEXT_PROMPT}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]
        )
        comp_image = self.client.chat.completions.create(
            model="o4-mini",
            messages=[{"role": "user", "content": [{"type": "text", "text": self.IMAGE_PROMPT}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]
        )
        return comp_text.choices[0].message.content, comp_image.choices[0].message.content

    def _save_section_to_db(self, section_id, page_id, text_info, image_info, vsm_vector):
        query = """
        MATCH (b:Book {contents_id: $contents_id})
        MERGE (bs:BookSection {contentssection_id: $section_id})
        SET bs.page_start = $page_id,
            bs.page_end = $page_id,
            bs.contents = $contents,
            bs.images = $images,
            bs.vsm = $vsm,
            bs.object_id = $object_id,
            bs.updated_at = datetime(),
            bs.is_pre_defined = false
        MERGE (bs)-[:PART_OF]->(b)
        """
        with self.driver.session() as session:
            session.run(query, contents_id=self.contents_id, section_id=section_id,
                        page_id=page_id, contents=text_info, images=image_info, vsm=vsm_vector,
                        object_id=self.object_id)

    def process_images(self, images):
        for i, image in enumerate(images, start=1):
            logger.info(f"  - [{self.__class__.__name__}] Processing Page {i}...")
            
            base64_image = self._encode_image(image)
            text_info, image_info = self._call_openai_vision(base64_image)
            vsm_vector = self.infra.get_embedding(text_info)
            
            section_id = f"{self.contents_id}_{i}_{i}"
            self._save_section_to_db(section_id, i, text_info, image_info, vsm_vector)
            
            self.post_process_page(section_id, text_info, vsm_vector)

    def post_process_page(self, section_id, text_info, vsm_vector):
        pass


class MathProcessor(BaseSubjectProcessor):
    TEXT_PROMPT = '''
        あなたの仕事は画像に含まれている文章を抽出してテキストで出力することです。
        以下の画像には数学の問題・解答が書かれています。この画像に載っているテキストを出力してください。
        - 問題は以下のフォーマットのように**$$で囲み**mathjax形式で出力すること。
        例：$$\\int_{}^{} f(x)\\\\,dx = F(b) - F(a)$$
        例：$$\\beta + \\gamma \\{} \\\\ \\alpha \\{}$$
        ただし、以下の点に注意すること。
        - mathjaxフォーマットの&は使わないこと。
        - \\text フォーマットは使わないこと。
        - <, >の２つの記号は、必ず"<\\," ">\\," という形で出力すること。
        - $という記号は、フォーマットで与えられている2組の$$を除き使用しないこと。
        以下のフォーマットで、XXXXに問題、YYYYに解答の過程、ZZZZに最終的な答えを挿入して、左揃えで答えること。
        [問題] \n
        $$\\begin{}{}{} XXXX \\end{}{}$$
        [解答の過程] \n
        $$\\begin{}{}{} YYYY \\end{}{}$$
        [最終的な答え] \n
        $$ZZZZ$$
        '''.format(r"{a}", r"{b}", r"text{の値から、}", r"text{を求める}", "{array", "}{l", "}", "{array", "}", "{array", "}{l", "}", "{array", "}")

    IMAGE_PROMPT = '''
        あなたの仕事は画像に含まれている図の説明をテキストで出力することです。
        以下の画像には数学の問題・解答が書かれています。
        - 辺の長さや座標など、図を描くために必要な情報をすべて出力すること。
        - 文章を書き起こさず、図のみを説明してください。
        - markdownは必要ありません。
    '''

    def __init__(self, contents_id, object_id, core_infra, math_estimator):
        super().__init__(contents_id, object_id, core_infra)
        self.math_estimator = math_estimator

    def post_process_page(self, section_id, text_info, vsm_vector):
        self.math_estimator.process_math_page(text_info, section_id, vsm_vector)

def setup_nltk():
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

class EnglishProcessor(BaseSubjectProcessor):
    def __init__(self, contents_id, object_id, core_infra):
        super().__init__(contents_id, object_id, core_infra)
        self.TOPIC_LIST = []
        self.POS_MAPPING = {}
        self._load_master_data_from_db()
        
        setup_nltk()
        self.tokenizer = TreebankWordTokenizer()
        self.lemmatizer = WordNetLemmatizer()
        self.topic_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        self.cefr_dict = {}
        cefr_csv_path = os.path.join(BASE_DIR, "cefr_wordlist.csv")
        if os.path.exists(cefr_csv_path):
            with open(cefr_csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    w = row.get("word", "").strip().lower()
                    p = row.get("pos", "").strip().upper()
                    l = row.get("level", "").strip()
                    if w and p:
                        self.cefr_dict[(w, p)] = l
        else:
            logger.warning(f"CEFR wordlist not found at {cefr_csv_path}. Using empty dictionary.")

    def _load_master_data_from_db(self):
        with self.driver.session() as session:
            query_topic = """
            MATCH (u:Unit)
            WHERE u.unit_id >= "220000001" AND u.unit_id <= "229999999"
            RETURN u.unit_name AS name
            """
            self.TOPIC_LIST = [record["name"] for record in session.run(query_topic)]

            query_pos = """
            MATCH (p:Property)
            WHERE p.property_id >= "020000001" AND p.property_id <= "020999999"
            RETURN p.property_name AS name
            """
            db_pos_list = [record["name"] for record in session.run(query_pos)]
            
            for pos_name in db_pos_list:
                pos_lower = pos_name.lower().strip()
                if pos_lower == "noun": self.POS_MAPPING["NOUN"] = pos_name
                elif pos_lower == "verb": self.POS_MAPPING["VERB"] = pos_name
                elif pos_lower == "adjective": self.POS_MAPPING["ADJ"] = pos_name
                elif pos_lower == "adverb": self.POS_MAPPING["ADV"] = pos_name

    def _convert_nltk_pos_to_wordnet(self, tag):
        if tag.startswith("N"): return "NOUN", wordnet.NOUN
        if tag.startswith("V"): return "VERB", wordnet.VERB
        if tag.startswith("J"): return "ADJ", wordnet.ADJ
        if tag.startswith("R"): return "ADV", wordnet.ADV
        return None

    def _extract_knowledge(self, text):
        tokens = self.tokenizer.tokenize(text)
        tokens = [token for token in tokens if token.isalpha()]

        if not tokens: return []

        tagged_words = pos_tag(tokens)
        knowledge_counter = Counter()

        for word, nltk_pos in tagged_words:
            converted_pos = self._convert_nltk_pos_to_wordnet(nltk_pos)
            if converted_pos is None: continue

            pos_type, wordnet_pos = converted_pos
            lemma = self.lemmatizer.lemmatize(word.lower(), pos=wordnet_pos)
            if not lemma: continue

            prop_pos_name = self.POS_MAPPING.get(pos_type)
            if prop_pos_name:
                knowledge_counter[(lemma, prop_pos_name)] += 1

        knowledgelists = [{"word": w, "pos": p, "count": c} for (w, p), c in knowledge_counter.items()]
        knowledgelists.sort(key=lambda x: (x["word"], x["pos"]))
        return knowledgelists

    def _get_next_concept_id(self, session):
        query = """
        MATCH (c:Concept)
        WHERE toInteger(c.concept_id) >= 500000001
        RETURN coalesce(max(toInteger(c.concept_id)) + 1, 500000001) AS next_id
        """
        result = session.run(query).single()
        return str(result["next_id"]) if result else "500000001"

    def _find_concept(self, session, word):
        query = """
        MATCH (c:Concept {concept_name: $word, subject: '英語'})
        RETURN c.concept_id AS concept_id
        LIMIT 1
        """
        record = session.run(query, word=word).single()
        return record["concept_id"] if record else None

    def _classify_topics(self, word, pos):
        topic_text = "\n".join(f"- {topic}" for topic in self.TOPIC_LIST)
        prompt = f"""
        You classify English vocabulary into semantic topics.
        Target word: {word}
        Part of speech: {pos}
        Select all topics that clearly apply to the meaning of this word from this list:
        {topic_text}
        Return ONLY a valid JSON array of topic names with exact casing.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.topic_model,
                messages=[{"role": "user", "content": prompt}]
            )
            output = response.choices[0].message.content.strip()
            if output.startswith("```"):
                output = output.replace("```json", "").replace("```", "").strip()
            topics = json.loads(output)
            return [t for t in topics if t in self.TOPIC_LIST]
        except Exception as exc:
            logger.error(f"Topic classification failed: {exc}")
            return []

    def _find_cefr_level(self, word, pos_name):
        inv_map = {v: k for k, v in self.POS_MAPPING.items()}
        pos_key = inv_map.get(pos_name, "")
        level = self.cefr_dict.get((word.lower(), pos_key), "Others")
        return level if level in ["A1", "A2", "B1", "B2"] else "Others"

    def _create_new_concept(self, session, word, pos_name):
        concept_id = self._get_next_concept_id(session)
        
        query_concept = """
        CREATE (c:Concept {
            concept_id: $concept_id,
            concept_name: $concept_name,
            subject: '英語',
            updated_at: datetime(),
            is_pre_defined: false
        })
        """
        session.run(query_concept, concept_id=concept_id, concept_name=word)

        query_pos = """
        MATCH (c:Concept {concept_id: $concept_id})
        MATCH (p:Property {property_name: $pos_name, subject: '英語', description: 'part_of_speech'})
        MERGE (c)-[:BELONGS_TO]->(p)
        """
        session.run(query_pos, concept_id=concept_id, pos_name=pos_name)

        cefr_level = self._find_cefr_level(word, pos_name)
        query_cefr = """
        MATCH (c:Concept {concept_id: $concept_id})
        MATCH (p:Property {property_name: $level, subject: '英語', description: 'difficulty'})
        MERGE (c)-[:BELONGS_TO]->(p)
        """
        session.run(query_cefr, concept_id=concept_id, level=cefr_level)

        topics = self._classify_topics(word, pos_name)
        query_unit = """
        MATCH (c:Concept {concept_id: $concept_id})
        MATCH (u:Unit {unit_name: $topic, subject: '英語'})
        MERGE (c)-[r:RELATED_TO]->(u)
        SET r.ratio = 1, r.rank = -1
        """
        for topic in topics:
            session.run(query_unit, concept_id=concept_id, topic=topic)

        return concept_id

    def _link_section_to_concept(self, session, section_id, concept_id, count):
        query = """
        MATCH (bs:BookSection {contentssection_id: $section_id})
        MATCH (c:Concept {concept_id: $concept_id})
        MERGE (bs)-[r:CONTAINS]->(c)
        ON CREATE SET r.num = $count
        ON MATCH SET r.num = r.num + $count
        """
        session.run(query, section_id=section_id, concept_id=concept_id, count=count)

    def post_process_page(self, section_id, text_info, vsm_vector):
        knowledgelists = self._extract_knowledge(text_info)
        
        if not knowledgelists:
            logger.info(f"  [English] No valid concepts found for section {section_id}.")
            return

        with self.driver.session() as session:
            for knowledge in knowledgelists:
                word = knowledge["word"]
                pos = knowledge["pos"]
                count = knowledge["count"]

                concept_id = self._find_concept(session, word)
                if concept_id is None:
                    concept_id = self._create_new_concept(session, word, pos_name=pos)
                    logger.info(f"  [English] [NEW CONCEPT] {word} ({pos}) → {concept_id}")

                self._link_section_to_concept(session, section_id, concept_id, count)
                logger.info(f"  [English] [LINKED] {word} → {section_id} (count: {count})")
            
            logger.info(f"  [English] Completed linking {len(knowledgelists)} concepts for section {section_id}.")

class JapaneseProcessor(BaseSubjectProcessor):
    TEXT_PROMPT = '''
        あなたは日本語教育の専門家です。提示された画像（問題集）を正確に読み取り、指示に従って語彙を抽出してください。
        「語彙の理解／読み／書きを問う問題」を見つけて、問われている語彙を問題ごとに順番に出力してください。
        漢字、ひらがな、カタカナは、一般的な表記に直してください。
        該当する問題がない場合や、問われている語彙がない場合、「該当なし」と出力してください。
    '''
    IMAGE_PROMPT = "あなたの仕事は画像に含まれている図の説明をテキストで出力することです。図がない場合はその旨を伝えてください。"

    def __init__(self, contents_id, object_id, core_infra):
        super().__init__(contents_id, object_id, core_infra)
        self.tokenizer = Tokenizer()

    def get_word_info(self, text: str):
        tokens = list(self.tokenizer.tokenize(text.strip()))
        if not tokens:
            return text, "名詞"

        for token in tokens:
            pos_main = token.part_of_speech.split(",")[0]
            if pos_main in ["動詞", "形容詞", "名詞", "副詞"]:
                base_form = token.base_form if token.base_form != "*" else token.surface
                return base_form, pos_main
        
        first = tokens[0]
        base_form = first.base_form if first.base_form != "*" else first.surface
        pos_main = first.part_of_speech.split(",")[0]
        return base_form, pos_main

    def clean_and_extract_words(self, llm_output: str):
        lines = llm_output.strip().split("\n")
        extracted = []

        for line in lines:
            cleaned_line = re.sub(r"^[0-9\.\s\-\*]+", "", line).strip()
            if cleaned_line and cleaned_line != "該当なし":
                word, pos = self.get_word_info(cleaned_line)
                extracted.append((word, pos))

        return extracted

    def extract_japanese_letters(self, text: str) -> list[str]:
        target_pattern = re.compile(r'[ぁ-んァ-ヶ一-龥々]')
        small_chars = set('ぁぃぅぇぉゃゅょァィゥェォャュョ')

        letters = []
        i = 0
        length = len(text)
        while i < length:
            char = text[i]
            if target_pattern.match(char):
                if i + 1 < length and text[i + 1] in small_chars:
                    letters.append(char + text[i + 1])
                    i += 2
                else:
                    letters.append(char)
                    i += 1
            else:
                i += 1
        return letters

    def _get_next_word_concept_id(self, session):
        query = """
        MATCH (c:Concept {description: '単語', subject: '国語'})
        WHERE toInteger(c.concept_id) >= 810000001
        RETURN coalesce(max(toInteger(c.concept_id)) + 1, 810000001) AS next_id
        """
        result = session.run(query).single()
        return str(result["next_id"]) if result else "810000001"

    def _find_word_concept(self, session, word):
        query = """
        MATCH (c:Concept {concept_name: $word, description: '単語', subject: '国語'})
        RETURN c.concept_id AS concept_id
        LIMIT 1
        """
        record = session.run(query, word=word).single()
        return record["concept_id"] if record else None

    def _create_word_concept(self, session, word, pos):
        concept_id = self._get_next_word_concept_id(session)
        
        query_concept = """
        CREATE (c:Concept {
            concept_id: $concept_id,
            concept_name: $concept_name,
            description: '単語',
            subject: '国語',
            updated_at: datetime(),
            is_pre_defined: false
        })
        """
        session.run(query_concept, concept_id=concept_id, concept_name=word)

        query_pos = """
        MATCH (c:Concept {concept_id: $concept_id})
        MATCH (p:Property {property_name: $pos, subject: '国語', description: 'part_of_speech'})
        MERGE (c)-[:BELONGS_TO]->(p)
        """
        session.run(query_pos, concept_id=concept_id, pos=pos)
        
        return concept_id

    def _link_section_to_word_concept(self, session, section_id, concept_id):
        query = """
        MATCH (bs:BookSection {contentssection_id: $section_id})
        MATCH (c:Concept {concept_id: $concept_id})
        MERGE (bs)-[r:CONTAINS]->(c)
        SET r.num = -1
        """
        session.run(query, section_id=section_id, concept_id=concept_id)

    def _process_and_link_letters_from_words(self, session, section_id, word_list):
        unique_letters = set()
        
        for word in word_list:
            letters = self.extract_japanese_letters(word)
            unique_letters.update(letters)

        if not unique_letters:
            return

        letter_data = [{"char": k} for k in unique_letters]

        query = """
        MATCH (bs:BookSection {contentssection_id: $section_id})
        UNWIND $letter_data AS data
        MATCH (c:Concept {concept_name: data.char, description: '文字', subject: '国語'})
        MERGE (bs)-[r:CONTAINS]->(c)
        SET r.num = -1
        """
        session.run(query, section_id=section_id, letter_data=letter_data)

    def post_process_page(self, section_id, text_info, vsm_vector, raw_text=""):
        with self.driver.session() as session:
            word_pos_list = self.clean_and_extract_words(text_info)
            
            if word_pos_list:
                extracted_words = []
                
                for word, pos in set(word_pos_list):
                    extracted_words.append(word)
                    
                    concept_id = self._find_word_concept(session, word)
                    if concept_id is None:
                        concept_id = self._create_word_concept(session, word, pos)
                        logger.info(f"  [Japanese] [NEW WORD] {word} ({pos}) → {concept_id}")
                    
                    self._link_section_to_word_concept(session, section_id, concept_id)

                self._process_and_link_letters_from_words(session, section_id, extracted_words)
                
                logger.info(f"  [Japanese] Linked words and constituent letters for section {section_id}.")
            else:
                logger.info(f"  [Japanese] No valid words found for section {section_id}.")


def get_processor(subject, contents_id, object_id, core_infra, math_estimator):
    """教科名から適切なプロセッサを生成して返すFactory関数"""
    if "数学" in subject or "算数" in subject:
        return MathProcessor(contents_id, object_id, core_infra, math_estimator)
    elif "英語" in subject:
        return EnglishProcessor(contents_id, object_id, core_infra)
    elif "国語" in subject:
        return JapaneseProcessor(contents_id, object_id, core_infra)
    else:
        return BaseSubjectProcessor(contents_id, object_id, core_infra)


# ==========================================
# 4. メイン処理・ユーティリティ
# ==========================================
def download_pdf_bytes(contents_id):
    url = content_info.build_pdf_url(contents_id)
    headers = content_info.build_auth_headers()
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()

def create_book_node(driver, contents_id, contents_name, subject, school_year):
    query = """
    MERGE (b:Book {contents_id: $contents_id})
    SET b.contents_name = $contents_name,
        b.subject = $subject,
        b.school_year = $school_year,
        b.updated_at = datetime(),
        b.is_pre_defined = false
    """
    with driver.session() as session:
        session.run(query, contents_id=contents_id, contents_name=contents_name,
                    subject=subject, school_year=school_year)

def process_single_content(contents_id, contents_name, course_name, object_id):
    logger.info(f"★ START Processing: {contents_name} (ID: {contents_id})")

    try:
        # メタデータの抽出
        match_subject = re.search(r'\[(.*?)\]', course_name)
        subject = match_subject.group(1) if match_subject else ""
        match_year = re.search(r'年度(.*?)年', course_name)
        school_year = (match_year.group(1) + "年") if match_year else ""
        
        logger.info(f"  - Meta: Subject={subject}, Year={school_year}")

        # Bookノードの作成
        create_book_node(global_infra.driver, contents_id, contents_name, subject, school_year)

        # PDFのダウンロードと画像化
        pdf_bytes = download_pdf_bytes(contents_id)
        
        # デバッグ用保存
        with open("/app/debug_downloaded.pdf", "wb") as f:
            f.write(pdf_bytes)
            
        images = convert_from_bytes(pdf_bytes, dpi=450, fmt='jpeg')
        logger.info(f"  - Converted {len(images)} pages.")

        # 教科に応じたプロセッサを取得し、一括処理を実行
        processor = get_processor(subject, contents_id, object_id, global_infra, global_math_estimator)
        processor.process_images(images)

        logger.info(f"✔ Completed: {contents_name}")

    except Exception as e:
        logger.error(f"❌ Failed to process {contents_id}: {e}", exc_info=True)


# ==========================================
# 5. モンキーパッチ & 起動
# ==========================================
def custom_handle_rows(rows):
    logger.info("Intercepted batch: Processing %s rows", len(rows))
    
    candidates = []
    for row in rows:
        # 新しいSELECT句に基づくインデックス
        # 0: QUERY_OPERATION = operation_name
        # 1: QUERY_CONTENTSID = contents_id
        # 2: QUERY_CONTENTSNAME = contents_name
        # 3: QUERY_COURSENAME = context_label
        # 4: QUERY_TIMESTAMP = timestamp
        # 5: QUERY_CONTENTSURL = object_id
        
        if row[0] != "REGISTER_CONTENTS": continue
        if not row[1]: continue

        candidates.append({
            "contents_id": row[1],
            "contents_name": row[2],
            "course_name": row[3],
            "object_id": row[5] # index 5 から object_id を取得
        })

    if not candidates: return

    verified_items = []
    def worker(item):
        try:
            content_info.check_pdf_endpoint(item["contents_id"])
            return item
        except Exception as e:
            logger.error(f"❌ PDFのアクセス検証に失敗しました ({item['contents_id']}): {e}")
            return None

    with ThreadPoolExecutor(max_workers=content_info.LEAF_MAX_WORKERS) as executor:
        for future in as_completed([executor.submit(worker, item) for item in candidates]):
            if res := future.result(): verified_items.append(res)

    for item in verified_items:
        process_single_content(
            contents_id=item["contents_id"], 
            contents_name=item["contents_name"],
            course_name=item["course_name"],
            object_id=item["object_id"] # object_id を渡す
        )

def main():
    global global_infra, global_math_estimator

    if not OPENAI_API_KEY:
        logger.error("Missing OPENAI_API_KEY")
        return

    logger.info("Initializing Core Infrastructure...")
    global_infra = CoreInfra(NEO4J_URL, NEO4J_AUTH, OPENAI_API_KEY)

    logger.info("Initializing Math Unit Estimator...")
    global_math_estimator = MathUnitEstimator(global_infra, REFERENCE_JSON_PATH, TFIDF_CSV_PATH)

    logger.info("Applying Monkey Patch to content_info...")
    content_info.handle_rows = custom_handle_rows

    logger.info("Starting monitoring loop...")
    try:
        content_info.main()
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        if global_infra:
            global_infra.close()

if __name__ == "__main__":
    main()