import os
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
from neo4j import GraphDatabase
from openai import OpenAI
from datetime import datetime

# ==========================================
# 設定
# ==========================================
NEO4J_URI = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MATH_UNIT_CSV = "math_unit.csv"
MATH_SUBUNIT_CSV = "math_subunit.csv"
MATH_KEYWORDS_CSV = "math_keywords.csv"
MATH_QUIZ_CSV = "math_quiz.csv"

# ==========================================
# ヘルパー関数
# ==========================================
def get_embedding(client, text):
    """テキストをベクトル化する関数"""
    text = text.replace("\n", " ")
    if not text: 
        return np.zeros(1536).tolist()
    try:
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"OpenAI Embedding Error: {e}")
        return np.zeros(1536).tolist()

def fetch_math_concepts(driver):
    """DBから数学のConceptを文字数の長い順に取得（部分一致の誤爆を防ぐため）"""
    query = "MATCH (c:Concept {subject: '数学'}) RETURN c.concept_name AS name"
    with driver.session() as session:
        result = session.run(query)
        # 文字列が長い順にソートして返す
        return sorted([record["name"] for record in result if record["name"]], key=len, reverse=True)

def extract_keywords_recursive(text, concepts):
    """テキストからキーワードを抽出し、出現回数をカウントする関数"""
    text_info_tmp = text
    extracted_counts = {}
    for concept_name in concepts:
        if not text_info_tmp: break
        count = text_info_tmp.count(concept_name)
        if count > 0:
            extracted_counts[concept_name] = count
            text_info_tmp = text_info_tmp.replace(concept_name, "")
    return extracted_counts

# ==========================================
# 各CSVの処理関数
# ==========================================
def process_math_units(driver):
    """math_unit.csv を読み込み、Unitの作成と学年への紐付けを行う"""
    if not os.path.exists(MATH_UNIT_CSV):
        print(f"❌ ファイルが見つかりません: {MATH_UNIT_CSV}")
        return

    df = pd.read_csv(MATH_UNIT_CSV, dtype=str)
    
    query = """
    MERGE (u:Unit {unit_id: $id})
    SET u.unit_name = $name,
        u.subject = "数学",
        u.is_pre_defined = true,
        u.updated_at = datetime()
    WITH u
    MATCH (p:Property {property_id: $grade, description: "grades"})
    MERGE (u)-[r:PART_OF]->(p)
    SET r.updated_at = datetime()
    """

    print(f"🚀 [1/4] {MATH_UNIT_CSV} の登録を開始します...")
    with driver.session() as session:
        for _, row in df.iterrows():
            u_id = str(row.get('id', '')).strip()
            u_name = str(row.get('name', '')).strip()
            grade_id = str(row.get('grade', '')).strip()
            
            if not u_id or u_id == 'nan': continue
            session.run(query, id=u_id, name=u_name, grade=grade_id)
    print(f"✅ {MATH_UNIT_CSV} の登録が完了しました。")

def process_math_subunits(driver):
    """math_subunit.csv を読み込み、Unit(小単元)の作成と親Unitへの紐付けを行う"""
    if not os.path.exists(MATH_SUBUNIT_CSV):
        print(f"❌ ファイルが見つかりません: {MATH_SUBUNIT_CSV}")
        return

    df = pd.read_csv(MATH_SUBUNIT_CSV, dtype=str)
    
    query = """
    MERGE (sub:Unit {unit_id: $id})
    SET sub.unit_name = $name,
        sub.subject = "数学",
        sub.updated_at = datetime(),
        sub.is_pre_defined = true
    WITH sub
    MATCH (parent:Unit {unit_id: $child_of})
    MERGE (sub)-[r:PART_OF]->(parent)
    SET r.updated_at = datetime()
    """

    print(f"🚀 [2/4] {MATH_SUBUNIT_CSV} の登録を開始します...")
    with driver.session() as session:
        for _, row in df.iterrows():
            sub_id = str(row.get('id', '')).strip()
            sub_name = str(row.get('name', '')).strip()
            parent_id = str(row.get('child_of', '')).strip()
            
            if not sub_id or sub_id == 'nan': continue
            session.run(query, id=sub_id, name=sub_name, child_of=parent_id)
    print(f"✅ {MATH_SUBUNIT_CSV} の登録が完了しました。")

def process_math_keywords(driver):
    """math_keywords.csv を読み込み、Concept(単語)の作成を行う"""
    if not os.path.exists(MATH_KEYWORDS_CSV):
        print(f"❌ ファイルが見つかりません: {MATH_KEYWORDS_CSV}")
        return

    df = pd.read_csv(MATH_KEYWORDS_CSV, dtype=str)
    
    query = """
    MERGE (c:Concept {concept_id: $id})
    SET c.concept_name = $name,
        c.description = "単語",
        c.subject = "数学",
        c.updated_at = datetime(),
        c.is_pre_defined = true
    """

    print(f"🚀 [3/4] {MATH_KEYWORDS_CSV} の登録を開始します...")
    with driver.session() as session:
        for _, row in df.iterrows():
            c_id = str(row.get('id', '')).strip()
            c_name = str(row.get('name', '')).strip()
            
            if not c_id or c_id == 'nan': continue
            session.run(query, id=c_id, name=c_name)
    print(f"✅ {MATH_KEYWORDS_CSV} の登録が完了しました。")

def process_math_quizzes(driver, client):
    """math_quiz.csv を読み込み、BookSectionとして登録し、単元・キーワードと紐付ける"""
    if not os.path.exists(MATH_QUIZ_CSV):
        print(f"❌ ファイルが見つかりません: {MATH_QUIZ_CSV}")
        return

    df = pd.read_csv(MATH_QUIZ_CSV, dtype=str)
    
    # 登録済みのキーワードを取得
    concepts = fetch_math_concepts(driver)
    
    # 教科書(textbook)用クエリ
    q_textbook = """
    MERGE (bs:BookSection {contentssection_id: $bs_id})
    SET bs.type = "textbook",
        bs.object_id = "",
        bs.answer_contentssection_id = "",
        bs.answer_object_id = "",
        bs.page_start = toInteger($page_s),
        bs.page_end = toInteger($page_e),
        bs.contents = $contents,
        bs.images = "",
        bs.vsm = $vsm,
        bs.subject = "数学",
        bs.updated_at = datetime(),
        bs.is_pre_defined = true
    """
    
    # 問題(question)用クエリ
    q_question = """
    MERGE (bs:BookSection {contentssection_id: $bs_id})
    SET bs.type = "question",
        bs.object_id = $object_id,
        bs.answer_contentssection_id = $ans_bs_id,
        bs.answer_object_id = $ans_object_id,
        bs.page_start = toInteger($page_s),
        bs.page_end = toInteger($page_e),
        bs.contents = $contents,
        bs.images = "",
        bs.vsm = $vsm,
        bs.subject = "数学",
        bs.updated_at = datetime(),
        bs.is_pre_defined = true
    """
    
    # 単元紐付けクエリ
    q_unit = """
    MATCH (bs:BookSection {contentssection_id: $bs_id})
    MATCH (u:Unit {unit_id: $sub_unit})
    MERGE (bs)-[r:RELATED_TO]->(u)
    SET r.ratio = 1.0, r.rank = 1, r.updated_at = datetime()
    """
    
    # キーワード紐付けクエリ
    q_concept = """
    MATCH (bs:BookSection {contentssection_id: $bs_id})
    UNWIND $keyword_data AS kw
    MATCH (c:Concept {concept_name: kw.name, subject: '数学'})
    MERGE (bs)-[r:CONTAINS]->(c)
    SET r.num = kw.count, r.updated_at = datetime()
    """

    print(f"🚀 [4/4] {MATH_QUIZ_CSV} の登録を開始します...")
    with driver.session() as session:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Math Quizzes"):
            # 値の取得とクレンジング
            contentsid = str(row.get('contentsid', '')).strip()
            if not contentsid or contentsid == 'nan': continue
            
            answer_id = str(row.get('answer_id', '')).strip()
            answer_id = "" if answer_id == 'nan' else answer_id
            
            page_s = str(row.get('page_s', '1')).strip()
            page_e = str(row.get('page_e', '1')).strip()
            page_qs = str(row.get('page_qs', '1')).strip()
            page_qe = str(row.get('page_qe', '1')).strip()
            
            # --- 1. bs_id の組み立て ---
            if not answer_id:
                # Textbookの場合
                bs_id = f"{contentsid}_{page_s}_{page_e}"
                ans_bs_id = ""
            else:
                # Questionの場合
                bs_id = f"{contentsid}_{page_qs}_{page_qe}"
                ans_bs_id = f"{answer_id}_{page_s}_{page_e}"

            # --- 2. DB既存チェック（存在すればスキップ） ---
            check_query = "MATCH (bs:BookSection {contentssection_id: $bs_id}) RETURN bs.contentssection_id LIMIT 1"
            if session.run(check_query, bs_id=bs_id).single() is not None:
                continue
            
            # --- 3. 未登録の場合のみ値を取得してVSM計算 ---
            contents = str(row.get('contents', '')).strip()
            contents = "" if contents == 'nan' else contents
            
            contents_view = str(row.get('contents_view', '')).strip()
            contents_view = "" if contents_view == 'nan' else contents_view
            
            answer_view = str(row.get('answer_view', '')).strip()
            answer_view = "" if answer_view == 'nan' else answer_view
            
            sub_unit = str(row.get('sub_unit', '')).strip()
            sub_unit = "" if sub_unit == 'nan' else sub_unit
            
            # VSMの処理（空の場合はOpenAIで生成）
            vsm_str = str(row.get('vsm', '')).strip()
            if not vsm_str or vsm_str == 'nan':
                vsm_vec = get_embedding(client, contents)
            else:
                try:
                    vsm_vec = ast.literal_eval(vsm_str)
                except (ValueError, SyntaxError):
                    vsm_vec = get_embedding(client, contents)
            
            # --- 4. BookSectionノードの登録 ---
            if not answer_id:
                session.run(q_textbook, bs_id=bs_id, page_s=page_s, page_e=page_e, 
                            contents=contents, vsm=vsm_vec)
            else:
                session.run(q_question, bs_id=bs_id, object_id=contents_view, 
                            ans_bs_id=ans_bs_id, ans_object_id=answer_view, 
                            page_s=page_s, page_e=page_e, contents=contents, vsm=vsm_vec)
            
            # --- 5. 小単元への紐付け (RELATED_TO) ---
            if sub_unit:
                session.run(q_unit, bs_id=bs_id, sub_unit=sub_unit)
            
            # --- 6. キーワードへの紐付け (CONTAINS) ---
            keywords_dict = extract_keywords_recursive(contents, concepts)
            if keywords_dict:
                kw_data = [{"name": k, "count": v} for k, v in keywords_dict.items()]
                session.run(q_concept, bs_id=bs_id, keyword_data=kw_data)

    print(f"✅ {MATH_QUIZ_CSV} の登録が完了しました。")

# ==========================================
# メイン処理
# ==========================================
def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        process_math_units(driver)
        process_math_subunits(driver)
        process_math_keywords(driver)
        process_math_quizzes(driver, client)
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
    finally:
        driver.close()
        print("🎉 すべての数学ノードの初期登録が完了しました！")

if __name__ == "__main__":
    main()