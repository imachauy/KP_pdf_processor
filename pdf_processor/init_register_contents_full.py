import os
import json
import pandas as pd
from neo4j import GraphDatabase
from openai import OpenAI
from tqdm import tqdm  # 進捗表示用

# ==========================================
# 設定
# ==========================================
# CSVファイル名
CSV_FILE = "contents_for_register.csv"

# 環境変数 (設定されていない場合はデフォルト値またはエラー)
NEO4J_URI = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ==========================================
# ヘルパー関数
# ==========================================
def get_embedding_as_text(client, text):
    """OpenAI APIでテキストをベクトル化し、JSON文字列として返す"""
    if not text:
        return "[]"
    text = text.replace("\n", " ")
    try:
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        vector_list = response.data[0].embedding
        return json.dumps(vector_list)
    except Exception as e:
        print(f"⚠️ Embedding Error: {e}")
        return "[]"

def fetch_master_data(driver):
    """Neo4jからUnitとConceptのマスタデータを取得する"""
    units = {}    # {unit_name: unit_id}
    concepts = {} # {concept_name: concept_id}

    with driver.session() as session:
        # Unit取得
        result_u = session.run("MATCH (u:Unit) RETURN u.unit_name, u.unit_id")
        for record in result_u:
            if record["u.unit_name"]:
                units[record["u.unit_name"]] = record["u.unit_id"]

        # Concept取得
        result_c = session.run("MATCH (c:Concept) RETURN c.concept_name, c.concept_id")
        for record in result_c:
            if record["c.concept_name"]:
                concepts[record["c.concept_name"]] = record["c.concept_id"]
    
    return units, concepts

# ==========================================
# メイン処理
# ==========================================
def main():
    # 1. 前準備
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY environment variable is not set.")
        return

    print(f"📖 Reading {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"❌ File not found: {CSV_FILE}")
        return

    client = OpenAI(api_key=OPENAI_API_KEY)
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    # 2. マスタデータのロード (Unit, Concept)
    print("📥 Loading Units and Concepts from Neo4j...")
    unit_map, concept_map = fetch_master_data(driver)
    print(f"   - Found {len(unit_map)} Units")
    print(f"   - Found {len(concept_map)} Concepts")

    # 3. 行ごとの処理
    print("🚀 Starting registration process...")
    
    # 登録用クエリの定義
    query_create_bs = """
    MERGE (bs:BookSection {contentssection_id: $bs_id})
    SET bs.contents_id = $c_id,
        bs.page_start = $p_start,
        bs.page_end = $p_end,
        bs.contents = $contents,
        bs.images = "",
        bs.vsm = $vsm
    """

    query_link_unit = """
    MATCH (bs:BookSection {contentssection_id: $bs_id})
    MATCH (u:Unit {unit_id: $u_id})
    MERGE (bs)-[:RELATED_TO {rank: 1}]->(u)
    """

    query_link_concept = """
    MATCH (bs:BookSection {contentssection_id: $bs_id})
    MATCH (c:Concept {concept_id: $c_id})
    MERGE (bs)-[:CONTAINS {num: $num}]->(c)
    """

    with driver.session() as session:
        # tqdmを使って進捗バーを表示
        for index, row in tqdm(df.iterrows(), total=len(df)):
            try:
                # --- A. データ準備 ---
                contents = row['contents']
                if pd.isna(contents): contents = ""
                
                c_id = row['contentsid']
                p_s = int(row['page_s'])
                p_e = int(row['page_e'])
                
                # ID生成: {contentsid}_{page_s}_{page_e}
                bs_id = f"{c_id}_{p_s}_{p_e}"
                
                # --- B. Embedding (VSM) ---
                # すでにCSVにvsmがあればそれを使う実装も可能ですが、
                # 今回は空(NaN)なのでAPIで生成します
                vsm_json = get_embedding_as_text(client, contents)

                # --- C. BookSectionノード作成 ---
                session.run(query_create_bs, 
                            bs_id=bs_id, c_id=c_id, p_start=p_s, p_end=p_e, 
                            contents=contents, vsm=vsm_json)

                # --- D. Unitとの紐付け (RELATED_TO) ---
                # sub_unit を優先し、なければ main_unit を使うロジック
                sub_u_name = row['sub_unit']
                main_u_name = row['main_unit']
                
                target_u_id = None
                
                # sub_unitの名前で検索
                if pd.notna(sub_u_name) and sub_u_name in unit_map:
                    target_u_id = unit_map[sub_u_name]
                # 見つからなければ main_unitの名前で検索
                elif pd.notna(main_u_name) and main_u_name in unit_map:
                    target_u_id = unit_map[main_u_name]
                
                if target_u_id:
                    session.run(query_link_unit, bs_id=bs_id, u_id=target_u_id)

                # --- E. Conceptとの紐付け (CONTAINS) ---
                # 全Conceptについて出現回数をカウント (単純な文字列一致)
                # 高速化のため、出現したものだけをリストアップしてまとめてクエリ実行しても良いですが、
                # ここでは確実性を重視してループ処理します
                
                for c_name, c_node_id in concept_map.items():
                    # Concept名が本文に含まれる回数をカウント
                    count = contents.count(c_name)
                    if count > 0:
                        session.run(query_link_concept, bs_id=bs_id, c_id=c_node_id, num=count)

            except Exception as e:
                print(f"⚠️ Error at row {index}: {e}")

    driver.close()
    print("✅ All done!")

if __name__ == "__main__":
    main()