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

# Neo4j接続設定
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# スクリプト自身のディレクトリパスを取得
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REFERENCE_JSON_PATH = os.path.join(BASE_DIR, "parent_contentpage.json")
TFIDF_CSV_PATH = os.path.join(BASE_DIR, "unit_tfidf_matrix_normalized.csv")

# グローバル変数としてEstimatorを保持
global_estimator = None

# ==========================================
# 0. クラス定義: PageUnitEstimator
# ==========================================
class PageUnitEstimator:
    def __init__(self, neo4j_url, neo4j_auth, openai_api_key, reference_json_path, tfidf_csv_path):
        """初期化: データベース接続と参照データのメモリへの読み込み"""
        self.driver = GraphDatabase.driver(neo4j_url, auth=neo4j_auth)
        self.client = OpenAI(api_key=openai_api_key)
        
        # 1. 教科書データ(Reference JSON)の読み込み
        logger.info(f"Loading reference data from {reference_json_path}...")
        try:
            with open(reference_json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            raw_data = []

        self.reference_records = []
        vsm_list = []

        # VSMのパースと有効データの抽出
        for record in raw_data:
            vsm_val = record.get('vsm')
            vec = None
            
            if isinstance(vsm_val, list):
                if len(vsm_val) > 0:
                    vec = np.array(vsm_val)
            elif isinstance(vsm_val, str) and vsm_val.strip():
                try:
                    parsed = ast.literal_eval(vsm_val)
                    if len(parsed) > 0:
                        vec = np.array(parsed)
                except:
                    pass
            
            if vec is not None:
                self.reference_records.append(record)
                vsm_list.append(vec)
        
        if vsm_list:
            self.reference_vsm_matrix = np.vstack(vsm_list)
        else:
            self.reference_vsm_matrix = np.array([])
        
        # 2. TF-IDF行列の読み込み
        logger.info(f"Loading TF-IDF matrix from {tfidf_csv_path}...")
        self.tfidf_data = {}
        self.tfidf_keywords = set()
        
        try:
            with open(tfidf_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    self.tfidf_keywords = set(reader.fieldnames) - {'unit_id'}
                
                for row in reader:
                    uid = row.get('unit_id')
                    if uid:
                        scores = {}
                        for k, v in row.items():
                            if k != 'unit_id':
                                try:
                                    val = float(v)
                                    if val > 0:
                                        scores[k] = val
                                except:
                                    pass
                        self.tfidf_data[str(uid)] = scores
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")

        # 3. Concept辞書のロード
        logger.info("Loading concepts from Neo4j...")
        self.concepts = self._fetch_concepts()

    def close(self):
        self.driver.close()

    def _fetch_concepts(self):
        query = """
        MATCH (c:Concept)
        WHERE toInteger(c.concept_id) > 989999999
        RETURN c.concept_name AS name
        """
        try:
            with self.driver.session() as session:
                result = session.run(query)
                concepts = [record["name"] for record in result]
            return sorted(concepts, key=len, reverse=True)
        except Exception as e:
            logger.error(f"Neo4j Error: {e}")
            return []

    def _extract_keywords_recursive(self, text):
        text_info_tmp = text
        extracted_counts = {}
        for concept_name in self.concepts:
            if not text_info_tmp: break
            count = text_info_tmp.count(concept_name)
            if count > 0:
                extracted_counts[concept_name] = count
                text_info_tmp = text_info_tmp.replace(concept_name, "")
        return extracted_counts

    def _get_embedding(self, text):
        text = text.replace("\n", " ")
        if not text: return np.zeros(1536)
        try:
            response = self.client.embeddings.create(
                input=[text],
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI Embedding Error: {e}")
            return np.zeros(1536)

    def process_page(self, text_info, contentssection_id, pre_calculated_vsm=None):
        keywords_counts = self._extract_keywords_recursive(text_info)
        detected_keywords = list(keywords_counts.keys())
        
        # ここから追加: 既存のConceptノードとBookSectionをリレーションで紐づける
        if keywords_counts:
            # キーワード名と出現回数のリストを作成
            keyword_data = [{"name": k, "count": v} for k, v in keywords_counts.items()]
            
            # UNWINDを使って一括で処理し、HAS_CONCEPTという線で結ぶクエリ
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
                    session.run(query_link_concepts, 
                                bs_id=contentssection_id, 
                                keyword_data=keyword_data)
                    logger.info(f"  [Estimator] Linked {len(keyword_data)} concepts to {contentssection_id}")
            except Exception as e:
                logger.error(f"Neo4j Concept Link Error: {e}")
        
        if pre_calculated_vsm is not None:
            vsm = np.array(pre_calculated_vsm).reshape(1, -1)
        else:
            vsm = np.array(self._get_embedding(text_info)).reshape(1, -1)

        valid_keywords = [kw for kw in detected_keywords if kw in self.tfidf_keywords]
        candidate_parent_unit_ids = set()
        
        if valid_keywords:
            for uid, scores_map in self.tfidf_data.items():
                total_score = sum(scores_map.get(kw, 0.0) for kw in valid_keywords)
                if total_score > 0:
                    candidate_parent_unit_ids.add(uid)

        if self.reference_vsm_matrix.size == 0:
            return []
            
        similarities = cosine_similarity(vsm, self.reference_vsm_matrix)[0]
        
        grouped_sims = {}

        for i, record in enumerate(self.reference_records):
            uid = str(record.get('unit_id', ''))
            
            if candidate_parent_unit_ids and uid not in candidate_parent_unit_ids:
                continue
                
            sub_id = str(record.get('subunit_id', ''))
            if not uid or not sub_id:
                continue

            sim = similarities[i]
            
            key = (uid, sub_id)
            if key not in grouped_sims:
                grouped_sims[key] = []
            grouped_sims[key].append(sim)

        if not grouped_sims:
            logger.warning(f"  [Estimator] Warning: No unit candidates found for {contentssection_id}")
            return []

        results = []
        for (parent_id, sub_id), sim_values in grouped_sims.items():
            sim_values.sort(reverse=True)
            top_n_values = sim_values[:3]
            
            if not top_n_values: continue
            score = median(top_n_values)
            
            results.append({
                "unit_id": parent_id,
                "subunit_id": sub_id,
                "score": score
            })
            
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)[:3]
        
        self._connect_to_neo4j(contentssection_id, sorted_results)
        return sorted_results

    def _connect_to_neo4j(self, contentssection_id, candidates):
        query = """
        MATCH (bs:BookSection {contentssection_id: $bs_id})
        MATCH (u:Unit {unit_id: $target_unit_id})
        MERGE (bs)-[r:RELATED_TO]->(u)
        SET r.ratio = $score, 
            r.rank = $rank
        """
        try:
            with self.driver.session() as session:
                for i, cand in enumerate(candidates, 1):
                    target_id = cand['subunit_id']
                    if target_id:
                        session.run(query, 
                                    bs_id=contentssection_id,
                                    target_unit_id=target_id,
                                    score=float(cand['score']),
                                    rank=i)
                        logger.info(f"  [Estimator] Connected {contentssection_id} -> SubUnit: {target_id} (Score: {cand['score']:.4f})")
        except Exception as e:
            logger.error(f"Neo4j Connect Error: {e}")


# ==========================================
# 2. 抽出・画像処理ヘルパー
# ==========================================
def encode_image(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    return base64_str

def extract_info_from_image_raw(pdf_one_page_image):
    client = global_estimator.client  # Estimatorのクライアントを再利用
    base64_image = encode_image(pdf_one_page_image)
    
    query_text = '''
        あなたの仕事は画像に含まれている文章を抽出してテキストで出力することです。
        以下の画像には数学の問題・解答が書かれています。この画像に載っているテキストを出力してください。
        - 解答の過程は、書いてある文字を出力すること。
        - 数式はmathjax形式を使って出力すること。数学記号はmathjaxで用いられている記号の表記を使うこと。
        - 分数の部分を明確に書くこと。例えば、\\frac14 ではなく、 \\frac{}{}{} と書く。 \\frac12x^2 ではなく、 \\frac{}{}{}x^2 と書く。
        - markdownは必要ありません。
        - 書いてある文字がすべて抽出できない場合は、抽出できた文字のみを出力すること。
        - 画像に含まれるテキストをそのまま出力することができない場合は、なぜできないのか説明してください。
    '''
    
    completion_text = client.chat.completions.create(
      model="o4-mini",
      messages=[
        {
          "role": "user",
          "content": [
            {"type": "text", "text": query_text},
            {"type": "image_url", "image_url": {"url":  f"data:image/jpeg;base64,{base64_image}"}},
          ],
        }
      ],
    )
    text_info = completion_text.choices[0].message.content

    query_image = '''
        あなたの仕事は画像に含まれている図の説明をテキストで出力することです。
        以下の画像には数学の問題・解答が書かれています。この画像に載っているテキストを出力してください。
        - 適切かつ一意に決まる表現を与えなければならない。
        - 辺の長さや座標など、図を描くために必要な情報をすべて出力すること。
        - 数式はmathjax形式を使って出力すること。
        - 分数の部分を明確に書くこと。例えば、\\frac14 ではなく、 \\frac{}{}{} と書く。 \\frac12x^2 ではなく、 \\frac{}{}{}x^2 と書く。
        - markdownは必要ありません。
        - 出力することができない場合は、なぜできないのか説明してください。
        - 文章を書き起こさず、図のみを説明してください。
    '''
        
    completion_image = client.chat.completions.create(
      model="o4-mini",
      messages=[
        {
          "role": "user",
          "content": [
            {"type": "text", "text": query_image},
            {"type": "image_url", "image_url": {"url":  f"data:image/jpeg;base64,{base64_image}"}},
          ],
        }
      ],
    )
    image_info = completion_image.choices[0].message.content

    return text_info, image_info

def download_pdf_bytes(contents_id):
    """APIからPDFをメモリ上にダウンロードする"""
    url = content_info.build_pdf_url(contents_id)
    headers = content_info.build_auth_headers()
    
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return response.read()
    except Exception as e:
        logger.error(f"Failed to download PDF for {contents_id}: {e}")
        raise


# ==========================================
# 3. メイン処理ロジック (1コンテンツ分)
# ==========================================
def process_single_content(contents_id, contents_name, course_name):
    """
    1つのコンテンツ(PDF)に対する処理フロー
    """
    logger.info(f"★ START Processing: {contents_name} (ID: {contents_id})")

    # 3-1. メタデータの抽出
    match_subject = re.search(r'\[(.*?)\]', course_name)
    subject = match_subject.group(1) if match_subject else ""

    match_year = re.search(r'年度(.*?)年', course_name)
    school_year = (match_year.group(1) + "年") if match_year else ""

    logger.info(f"  - Meta: Subject={subject}, Year={school_year}")

    try:
        # 3-2. Bookノードの作成
        query_create_book = """
        MERGE (b:Book {contents_id: $contents_id})
        SET b.contents_name = $contents_name,
            b.subject = $subject,
            b.school_year = $school_year,
            b.updated_at = datetime()
        """
        with global_estimator.driver.session() as session:
            session.run(query_create_book, 
                        contents_id=contents_id, 
                        contents_name=contents_name,
                        subject=subject, 
                        school_year=school_year)
            logger.info("  - Book node merged/updated.")

        # 3-3. PDFダウンロードと画像変換
        pdf_bytes = download_pdf_bytes(contents_id)
        logger.info("  - PDF downloaded. Converting to images (450dpi)...")

        # ====== ▼▼ デバッグ用に追加 ▼▼ ======
        debug_pdf_path = f"/app/debug_downloaded.pdf"
        with open(debug_pdf_path, "wb") as f:
            f.write(pdf_bytes)
        logger.info(f"  - [DEBUG] ダウンロードしたPDFのサイズ: {len(pdf_bytes)} bytes")
        # ====== ▲▲ デバッグ用に追加 ▲▲ ======
        
        # PDFをメモリから読み込んで画像変換 (popplerが必要です)
        images = convert_from_bytes(pdf_bytes, dpi=450, fmt='jpeg')
        logger.info(f"  - Converted {len(images)} pages.")

        # 3-4. 各ページ処理ループ
        query_create_section = """
        MATCH (b:Book {contents_id: $contents_id})
        MERGE (bs:BookSection {contentssection_id: $contentssection_id})
        SET bs.contents_id = $contents_id,
            bs.page_start = $page_start,
            bs.page_end = $page_end,
            bs.contents = $contents,
            bs.images = $images,
            bs.vsm = $vsm
        MERGE (bs)-[:PART_OF]->(b)
        """

        with global_estimator.driver.session() as session:
            for i, image in enumerate(images, start=1):
                page_id = i
                logger.info(f"  - Processing Page {page_id}...")
                
                # OpenAI Vision Extraction
                text_info, image_info = extract_info_from_image_raw(image)
                
                # VSM Calculation
                text_for_embedding = text_info.replace("\n", " ")
                vsm_vector = []
                if text_for_embedding:
                    try:
                        resp = global_estimator.client.embeddings.create(input=[text_for_embedding], model="text-embedding-3-small")
                        vsm_vector = resp.data[0].embedding
                    except Exception as e:
                        logger.error(f"  Embedding Error on page {page_id}: {e}")

                # ID生成 (ご指定のフォーマット)
                contentssection_id = f"{contents_id}_{page_id}_{page_id}"
                
                # Neo4j登録
                session.run(query_create_section,
                            contentssection_id=contentssection_id,
                            contents_id=contents_id,
                            page_start=page_id,
                            page_end=page_id,
                            contents=text_info,
                            images=image_info,
                            vsm=vsm_vector)
                
                # 単元推定実行
                global_estimator.process_page(text_info, contentssection_id, pre_calculated_vsm=vsm_vector)

        logger.info(f"✔ Completed: {contents_name}")

    except Exception as e:
        logger.error(f"❌ Failed to process {contents_id}: {e}", exc_info=True)


# ==========================================
# 4. content_info.py へのモンキーパッチ
# ==========================================
def custom_handle_rows(rows):
    """
    content_info.py の handle_rows を乗っ取り、
    検証OKだったものを process_single_content に流す
    """
    logger.info("Intercepted batch: Processing %s rows", len(rows))
    
    candidates = []
    for row in rows:
        if row[content_info.COL_OPERATION_NAME] != "REGISTER_CONTENTS":
            continue
        c_id = row[content_info.COL_CONTENTS_ID]
        if not c_id: continue

        item = {
            "contents_id": c_id,
            "contents_name": row[content_info.COL_CONTENTS_NAME],
            "context_label": row[content_info.COL_CONTEXT_LABEL]
        }
        candidates.append(item)

    if not candidates:
        return

    # PDFが存在するか検証 (並列)
    verified_items = []
    def worker(item):
        try:
            content_info.check_pdf_endpoint(item["contents_id"])
            return item
        except Exception as e:
            logger.error(f"❌ PDFのアクセス検証に失敗しました ({item['contents_id']}): {e}")
            return None

    with ThreadPoolExecutor(max_workers=content_info.LEAF_MAX_WORKERS) as executor:
        futures = [executor.submit(worker, item) for item in candidates]
        for future in as_completed(futures):
            res = future.result()
            if res:
                verified_items.append(res)

    # 重い処理なので1件ずつ順次実行
    for item in verified_items:
        process_single_content(
            contents_id=item["contents_id"],
            contents_name=item["contents_name"],
            course_name=item["context_label"] # context_labelをcourse_nameとして利用
        )

# ==========================================
# 5. メイン実行部
# ==========================================
def main():
    global global_estimator

    if not OPENAI_API_KEY:
        logger.error("Missing OPENAI_API_KEY")
        return

    # 1. 重いクラスを初期化 (初回のみ)
    logger.info("Initializing PageUnitEstimator...")
    global_estimator = PageUnitEstimator(
        neo4j_url=NEO4J_URL,
        neo4j_auth=NEO4J_AUTH,
        openai_api_key=OPENAI_API_KEY,
        reference_json_path=REFERENCE_JSON_PATH,
        tfidf_csv_path=TFIDF_CSV_PATH
    )

    # 2. モンキーパッチ適用
    logger.info("Applying Monkey Patch to content_info...")
    content_info.handle_rows = custom_handle_rows

    # 3. 監視ループ開始
    logger.info("Starting monitoring loop...")
    try:
        content_info.main()
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        if global_estimator:
            global_estimator.close()

if __name__ == "__main__":
    main()