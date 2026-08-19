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
from collections import Counter
from datetime import datetime, timezone

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
        # キーワード抽出
        text_info_tmp = text_info
        extracted_counts = {}
        for concept_name in self.concepts:
            if not text_info_tmp: break
            count = text_info_tmp.count(concept_name)
            if count > 0:
                extracted_counts[concept_name] = count
                text_info_tmp = text_info_tmp.replace(concept_name, "")
                
        # Neo4jにキーワードリンク
        if extracted_counts:
            keyword_data = [{"name": k, "count": v} for k, v in extracted_counts.items()]
            query_link_concepts = """
            MATCH (bs:BookSection {contentssection_id: $bs_id})
            UNWIND $keyword_data AS kw
            MATCH (c:Concept {concept_name: kw.name})
            WHERE toInteger(c.concept_id) > 989999999
            MERGE (bs)-[r:HAS_CONCEPT]->(c)
            SET r.count = kw.count
            """
            try:
                with self.driver.session() as session:
                    session.run(query_link_concepts, bs_id=contentssection_id, keyword_data=keyword_data)
            except Exception as e:
                logger.error(f"Neo4j Concept Link Error: {e}")

        # 単元推定
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
        
        # 単元リンク
        query_connect = """
        MATCH (bs:BookSection {contentssection_id: $bs_id})
        MATCH (u:Unit {unit_id: $target_unit_id})
        MERGE (bs)-[r:RELATED_TO]->(u)
        SET r.ratio = $score, r.rank = $rank
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
# 3. 教科別プロセッサ (Strategy Pattern)
# ==========================================
class BaseSubjectProcessor:
    """各教科の処理の土台となるクラス。共通ループやDB保存を担当。"""
    
    TEXT_PROMPT = "あなたの仕事は画像に含まれている文章を抽出してテキストで出力することです。"
    IMAGE_PROMPT = "あなたの仕事は画像に含まれている図の説明をテキストで出力することです。"

    def __init__(self, contents_id, core_infra):
        self.contents_id = contents_id
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
            bs.vsm = $vsm
        MERGE (bs)-[:PART_OF]->(b)
        """
        with self.driver.session() as session:
            session.run(query, contents_id=self.contents_id, section_id=section_id,
                        page_id=page_id, contents=text_info, images=image_info, vsm=vsm_vector)

    def process_images(self, images):
        """全教科共通の処理フロー（抽出 → ベクトル化 → 保存 → 固有処理）"""
        for i, image in enumerate(images, start=1):
            logger.info(f"  - [{self.__class__.__name__}] Processing Page {i}...")
            
            base64_image = self._encode_image(image)
            text_info, image_info = self._call_openai_vision(base64_image)
            vsm_vector = self.infra.get_embedding(text_info)
            
            section_id = f"{self.contents_id}_{i}_{i}"
            self._save_section_to_db(section_id, i, text_info, image_info, vsm_vector)
            
            # 各教科固有の事後処理を呼び出し
            self.post_process_page(section_id, text_info, vsm_vector)

    def post_process_page(self, section_id, text_info, vsm_vector):
        """教科ごとに上書きする事後処理"""
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
        $$ \\begin{}{}{} XXXX \\end{}{} $$
        [解答の過程] \n
        $$ \\begin{}{}{} YYYY \\end{}{} $$
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

    def __init__(self, contents_id, core_infra, math_estimator):
        super().__init__(contents_id, core_infra)
        self.math_estimator = math_estimator

    def post_process_page(self, section_id, text_info, vsm_vector):
        # 数学専用の単元推定ロジックを呼び出す
        self.math_estimator.process_math_page(text_info, section_id, vsm_vector)

# NLTKの辞書データを初回のみ安全にダウンロードするためのヘルパー関数
def setup_nltk():
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

class EnglishProcessor(BaseSubjectProcessor):

    # Neo4j Schema Settings
    POS_LABEL = "POS"
    POS_NAME_PROPERTY = "pos_name"
    TOPIC_LABEL = "Topic"
    TOPIC_NAME_PROPERTY = "topic_name"
    CEFR_LABEL = "CEFR"
    CEFR_LEVEL_PROPERTY = "level"

    TOPIC_LIST = [
        "animals", "appearance", "communication", "culture",
        "food and drink", "functions", "health", "homes and buildings",
        "leisure", "notions", "people", "politics and society",
        "science and technology", "sport", "the natural world",
        "time and space", "travel", "work and business",
    ]

    def __init__(self, contents_id, core_infra):
        super().__init__(contents_id, core_infra)
        
        setup_nltk()
        self.tokenizer = TreebankWordTokenizer()
        self.lemmatizer = WordNetLemmatizer()

        # Topic分類用のモデル名
        self.topic_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini") # 利用可能なモデル名に調整してください

        # CEFR-J wordlist のロード (pandasを使わず辞書化)
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

    # ============================================================
    # 抽出・構文解析
    # ============================================================
    def _convert_nltk_pos_to_wordnet(self, tag):
        if tag.startswith("N"): return "NOUN", wordnet.NOUN
        if tag.startswith("V"): return "VERB", wordnet.VERB
        if tag.startswith("J"): return "ADJ", wordnet.ADJ
        if tag.startswith("R"): return "ADV", wordnet.ADV
        return None

    def _extract_knowledge(self, text):
        tokens = self.tokenizer.tokenize(text)
        tokens = [token for token in tokens if token.isalpha()]

        if not tokens:
            return []

        tagged_words = pos_tag(tokens)
        knowledge_counter = Counter()

        for word, nltk_pos in tagged_words:
            converted_pos = self._convert_nltk_pos_to_wordnet(nltk_pos)
            if converted_pos is None:
                continue

            pos_name, wordnet_pos = converted_pos
            lemma = self.lemmatizer.lemmatize(word.lower(), pos=wordnet_pos)
            if not lemma:
                continue

            knowledge_counter[(lemma, pos_name)] += 1

        knowledgelists = [
            {"word": w, "pos": p, "count": c} 
            for (w, p), c in knowledge_counter.items()
        ]
        knowledgelists.sort(key=lambda x: (x["word"], x["pos"]))

        return knowledgelists

    # ============================================================
    # 既存Conceptの検索
    # ============================================================
    def _find_concept(self, session, word, pos):
        query = f"""
        MATCH (p:{self.POS_LABEL})-[:CONTAINS]->(c:Concept)
        WHERE toLower(c.concept_name) = toLower($word)
          AND toUpper(p.{self.POS_NAME_PROPERTY}) = toUpper($pos)
        RETURN c.concept_id AS concept_id
        LIMIT 1
        """
        record = session.run(query, word=word, pos=pos).single()
        return record["concept_id"] if record else None

    # ============================================================
    # 新規Conceptの作成プロセス
    # ============================================================
    def _classify_topics(self, word, pos):
        topic_text = "\n".join(f"- {topic}" for topic in self.TOPIC_LIST)
        prompt = f"""
        You classify English vocabulary into semantic topics.

        Target word:
        {word}

        Part of speech:
        {pos}

        Select all topics that clearly apply to the meaning of this word.

        A word may belong to:
        - no topic,
        - one topic,
        - or multiple topics.

        Do not select a topic merely because it has a weak or indirect relationship
        with the word.

        You must choose topics only from this list:

        {topic_text}

        Return ONLY a valid JSON array of topic names.
        Example:
        ["health", "people"]

        If none clearly applies:
        []

        Do not include explanations.
        """
        response = self.client.chat.completions.create(
            model=self.topic_model,
            messages=[{"role": "user", "content": prompt}]
        )
        output = response.choices[0].message.content.strip()

        if output.startswith("```"):
            output = output.replace("```json", "").replace("```", "").strip()

        try:
            topics = json.loads(output)
        except json.JSONDecodeError as exc:
            logger.error(f"Invalid JSON returned by LLM: {output}")
            return []

        return [t for t in topics if t in self.TOPIC_LIST]

    def _find_cefr_level(self, word, pos):
        # 辞書を使って一瞬で検索。見つからなければ "else" を返す
        key = (word.lower(), pos.upper())
        return self.cefr_dict.get(key, "else")

    def _create_newconcept(self, session, word, pos):
        concept_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # 1. CREATE Concept
        query_concept = """
        CREATE (c:Concept {
            concept_id: $concept_id,
            concept_name: $concept_name,
            description: "",
            author: "system_admin",
            created_at: $created_at
        })
        """
        session.run(query_concept, concept_id=concept_id, concept_name=word, created_at=created_at)

        # 2. Add POS Relation
        query_pos = f"""
        MATCH (p:{self.POS_LABEL})
        WHERE toUpper(p.{self.POS_NAME_PROPERTY}) = toUpper($pos)
        MATCH (c:Concept {{concept_id: $concept_id}})
        CREATE (p)-[:CONTAINS {{num: 1}}]->(c)
        RETURN count(p) AS created
        """
        if session.run(query_pos, concept_id=concept_id, pos=pos).single()["created"] == 0:
            logger.warning(f"POS node not found: {pos}")

        # 3. Add CEFR Relation
        cefr_level = self._find_cefr_level(word, pos)
        query_cefr = f"""
        MATCH (cefr:{self.CEFR_LABEL})
        WHERE toLower(cefr.{self.CEFR_LEVEL_PROPERTY}) = toLower($cefr_level)
        MATCH (c:Concept {{concept_id: $concept_id}})
        CREATE (cefr)-[:CONTAINS {{num: 1}}]->(c)
        RETURN count(cefr) AS created
        """
        if session.run(query_cefr, concept_id=concept_id, cefr_level=cefr_level).single()["created"] == 0:
            logger.warning(f"CEFR node not found: {cefr_level}")

        # 4. Add Topic Relation (LLM Classification)
        topics = self._classify_topics(word, pos)
        for topic in topics:
            query_topic = f"""
            MATCH (t:{self.TOPIC_LABEL})
            WHERE toLower(t.{self.TOPIC_NAME_PROPERTY}) = toLower($topic)
            MATCH (c:Concept {{concept_id: $concept_id}})
            CREATE (t)-[:CONTAINS {{num: 1}}]->(c)
            RETURN count(t) AS created
            """
            if session.run(query_topic, concept_id=concept_id, topic=topic).single()["created"] == 0:
                logger.warning(f"Topic node not found: {topic}")

        return concept_id

    # ============================================================
    # Section(Unit)とConceptの紐付け
    # ============================================================
    def _create_section_concept_relationship(self, session, section_id, concept_id, count):
        query = """
        MATCH (bs:BookSection {contentssection_id: $section_id})
        MATCH (c:Concept {concept_id: $concept_id})
        CREATE (bs)-[:CONTAINS {num: $count}]->(c)
        """
        session.run(query, section_id=section_id, concept_id=concept_id, count=count)

    # ============================================================
    # メインとなる事後処理 (post_process_page)
    # ============================================================
    def post_process_page(self, section_id, text_info, vsm_vector):
        knowledgelists = self._extract_knowledge(text_info)
        if not knowledgelists:
            return

        with self.driver.session() as session:
            for knowledge in knowledgelists:
                word = knowledge["word"]
                pos = knowledge["pos"]
                count = knowledge["count"]

                # 既存Conceptの検索
                concept_id = self._find_concept(session, word, pos)

                # 新規Conceptの作成
                if concept_id is None:
                    concept_id = self._create_newconcept(session, word, pos)
                    logger.info(f"  [English] [NEW CONCEPT] {word} / {pos} → {concept_id}")

                # BookSection(Page) と Concept を繋ぐ
                self._create_section_concept_relationship(session, section_id, concept_id, count)


class JapaneseProcessor(BaseSubjectProcessor):
    TEXT_PROMPT = '''
        あなたは日本語教育の専門家です。提示された画像（問題集）を正確に読み取り、指示に従って語彙を抽出してください。
        「語彙の理解／読み／書きを問う問題」を見つけて、問われている語彙を問題ごとに順番に出力してください。
        漢字、ひらがな、カタカナは、一般的な表記に直してください。
        該当する問題がない場合や、問われている語彙がない場合、「該当なし」と出力してください。
    '''
    IMAGE_PROMPT = "あなたの仕事は画像に含まれている図の説明をテキストで出力することです。図がない場合はその旨を伝えてください。"

    def __init__(self, contents_id, core_infra):
        super().__init__(contents_id, core_infra)
        # 形態素解析器 Janome の初期化
        self.tokenizer = Tokenizer()

    # ============================================================
    # 1. 抽出・構文解析 (Concept/語彙用)
    # ============================================================
    def lemmatize_word(self, text: str) -> str:
        """テキストから最初に見つかった動詞の基本形（辞書形）を取り出す"""
        tokens = list(self.tokenizer.tokenize(text.strip()))
        if not tokens:
            return text

        for token in tokens:
            # 品詞が「動詞」であるものを探す
            part_of_speech = token.part_of_speech.split(",")[0]
            if part_of_speech == "動詞":
                # 動詞の基本形（base_form）を返す（例: 「乗る」）
                return token.base_form if token.base_form != "*" else token.surface

        # 動詞が見つからない場合は形態素解析結果の先頭の基本形を返す
        first_token = tokens[0]
        return first_token.base_form if first_token.base_form != "*" else first_token.surface

    def clean_and_lemmatize_llm_output(self, llm_output: str) -> list[str]:
        """LLMが出力したテキスト（1. 通う 2. 医学...）から番号を取り除き、基本形に揃える"""
        lines = llm_output.strip().split("\n")
        cleaned_words = []

        for line in lines:
            # 先頭の数字や記号（「1. 」「- 」など）を除去
            cleaned_line = re.sub(r"^[0-9\.\s\-\*]+", "", line).strip()

            if cleaned_line and cleaned_line != "該当なし":
                # 形態素解析で基本形に変換
                lemmatized = self.lemmatize_word(cleaned_line)
                cleaned_words.append(lemmatized)

        return cleaned_words

    # ============================================================
    # 2. Neo4j 操作 (Concept/語彙用)
    # ============================================================
    def _find_concept(self, session, word):
        """Neo4jから既存のConceptを検索する"""
        query = """
        MATCH (c:Concept)
        WHERE c.concept_name = $word
        RETURN c.concept_id AS concept_id
        LIMIT 1
        """
        record = session.run(query, word=word).single()
        return record["concept_id"] if record else None

    def _create_concept(self, session, word):
        """新しいConceptノードを作成する"""
        concept_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        query = """
        CREATE (c:Concept {
            concept_id: $concept_id,
            concept_name: $concept_name,
            description: "",
            author: "system_admin",
            created_at: $created_at
        })
        """
        session.run(query, concept_id=concept_id, concept_name=word, created_at=created_at)
        return concept_id

    def _create_section_concept_relationship(self, session, section_id, concept_id, count):
        """BookSection(ページ)とConceptを出現回数付きで紐づける"""
        query = """
        MATCH (bs:BookSection {contentssection_id: $section_id})
        MATCH (c:Concept {concept_id: $concept_id})
        CREATE (bs)-[:CONTAINS {num: $count}]->(c)
        """
        session.run(query, section_id=section_id, concept_id=concept_id, count=count)

    # ============================================================
    # 3. 文字抽出・Neo4j 操作 (Letter用)
    # ============================================================
    def extract_japanese_letters(self, text: str) -> list[str]:
        """テキストから漢字・ひらがな・カタカナを抽出し、拗音は2文字セットにする"""
        # 抽出対象の文字（ひらがな、カタカナ、漢字、々）
        target_pattern = re.compile(r'[ぁ-んァ-ヶ一-龥々]')
        # 拗音など、前の文字と結合する小書き文字
        small_chars = set('ぁぃぅぇぉゃゅょァィゥェォャュョ')

        letters = []
        i = 0
        length = len(text)

        while i < length:
            char = text[i]
            if target_pattern.match(char):
                # 次の文字が小書き文字なら結合して2文字にする
                if i + 1 < length and text[i + 1] in small_chars:
                    letters.append(char + text[i + 1])
                    i += 2
                else:
                    letters.append(char)
                    i += 1
            else:
                i += 1
        return letters

    def save_letters_to_neo4j(self, session, section_id: str, raw_text: str):
        """テキストから文字を抽出し、LetterノードとしてBookSectionに紐づける"""
        letters = self.extract_japanese_letters(raw_text)
        if not letters:
            return

        letter_counts = Counter(letters)
        letter_data = [{"letter": k, "count": v} for k, v in letter_counts.items()]

        query = """
        MATCH (bs:BookSection {contentssection_id: $section_id})
        UNWIND $letter_data AS data
        
        // Letterノードが存在しなければ作成
        MERGE (l:Letter {character: data.letter})
        
        // BookSection と Letter を結ぶ
        MERGE (bs)-[r:CONTAINS_LETTER]->(l)
        ON CREATE SET r.num = data.count
        ON MATCH SET r.num = r.num + data.count
        """
        session.run(query, section_id=section_id, letter_data=letter_data)

    # ============================================================
    # 4. メインとなる事後処理
    # ============================================================
    def post_process_page(self, section_id, text_info, vsm_vector, raw_text=""):
        """
        LLMが抽出した語彙(text_info) と PDFの生テキスト(raw_text) を受け取り、
        Concept(語彙)ノードと Letter(文字)ノードを作成・紐づける
        """
        
        # --- A. 語彙(Concept)の処理 ---
        word_list = self.clean_and_lemmatize_llm_output(text_info)
        
        with self.driver.session() as session:
            if word_list:
                word_counts = Counter(word_list)
                for word, count in word_counts.items():
                    concept_id = self._find_concept(session, word)
                    if concept_id is None:
                        concept_id = self._create_concept(session, word)
                        logger.info(f"  [Japanese] [NEW CONCEPT] {word} → {concept_id}")
                    
                    self._create_section_concept_relationship(session, section_id, concept_id, count)
            else:
                logger.info(f"  [Japanese] No valid words found for section {section_id}.")

            # --- B. 文字(Letter)の処理 ---
            # PDFから直接抽出した生テキスト(raw_text)が渡されている場合のみ実行
            if raw_text:
                self.save_letters_to_neo4j(session, section_id, raw_text)
                logger.info(f"  [Japanese] Saved letters for section {section_id}.")


def get_processor(subject, contents_id, core_infra, math_estimator):
    """教科名から適切なプロセッサを生成して返すFactory関数"""
    if "数学" in subject or "算数" in subject:
        return MathProcessor(contents_id, core_infra, math_estimator)
    elif "英語" in subject:
        return EnglishProcessor(contents_id, core_infra)
    elif "国語" in subject:
        return JapaneseProcessor(contents_id, core_infra)
    else:
        return BaseSubjectProcessor(contents_id, core_infra)


# ==========================================
# 4. メイン処理・ユーティリティ
# ==========================================
def download_pdf_bytes(contents_id):
    url = content_info.build_pdf_url(contents_id)
    headers = content_info.build_auth_headers()
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()

def create_book_node(driver, contents_id, contents_name, subject, school_year, school_id):
    query = """
    MERGE (b:Book {contents_id: $contents_id})
    SET b.contents_name = $contents_name,
        b.subject = $subject,
        b.school_year = $school_year,
        b.school_id = $school_id,
        b.updated_at = datetime()
    """
    with driver.session() as session:
        session.run(query, contents_id=contents_id, contents_name=contents_name,
                    subject=subject, school_year=school_year, school_id=school_id)

def process_single_content(contents_id, school_id, contents_name, course_name):
    logger.info(f"★ START Processing: {contents_name} (ID: {contents_id})")

    try:
        # メタデータの抽出
        match_subject = re.search(r'\[(.*?)\]', course_name)
        subject = match_subject.group(1) if match_subject else ""
        match_year = re.search(r'年度(.*?)年', course_name)
        school_year = (match_year.group(1) + "年") if match_year else ""
        
        logger.info(f"  - Meta: Subject={subject}, Year={school_year}")

        # Bookノードの作成
        create_book_node(global_infra.driver, contents_id, contents_name, subject, school_year, school_id)

        # PDFのダウンロードと画像化
        pdf_bytes = download_pdf_bytes(contents_id)
        
        # デバッグ用保存
        with open("/app/debug_downloaded.pdf", "wb") as f:
            f.write(pdf_bytes)
            
        images = convert_from_bytes(pdf_bytes, dpi=450, fmt='jpeg')
        logger.info(f"  - Converted {len(images)} pages.")

        # 教科に応じたプロセッサを取得し、一括処理を実行
        processor = get_processor(subject, contents_id, global_infra, global_math_estimator)
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
        if row[content_info.COL_OPERATION_NAME] != "REGISTER_CONTENTS": continue
        if not row[content_info.COL_CONTENTS_ID]: continue

        candidates.append({
            "contents_id": row[content_info.COL_CONTENTS_ID],
            "contents_name": row[content_info.COL_CONTENTS_NAME],
            "school_id": row[content_info.COL_SCHOOL_ID],
            "context_label": row[content_info.COL_CONTEXT_LABEL]
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
            school_id=item["school_id"], 
            contents_name=item["contents_name"],
            course_name=item["context_label"]
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