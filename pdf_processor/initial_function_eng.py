import os
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm
from datetime import datetime

CSV_FILE = "eng_word_list.csv"
NEO4J_URI = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))

# マスタデータ定義
UNIT_DATA = [
    ("220000001", "Animals"), ("220000002", "Appearance"), ("220000003", "Communication"),
    ("220000004", "Culture"), ("220000005", "Food and drink"), ("220000006", "Functions"),
    ("220000007", "Health"), ("220000008", "Homes and buildings"), ("220000009", "Leisure"),
    ("220000010", "Notions"), ("220000011", "People"), ("220000012", "Politics and society"),
    ("220000013", "Science and technology"), ("220000014", "Sport"), ("220000015", "The natural world"),
    ("220000016", "Time and space"), ("220000017", "Travel"), ("220000018", "Work and business")
]

PROPERTY_DATA = [
    ("021000001", "A1"), ("021000002", "A2"), ("021000003", "B1"), ("021000004", "B2"), ("021000005", "Others"),
    ("020000001", "adjective"), ("020000002", "adverb"), ("020000003", "noun"), ("020000004", "verb")
]

def check_already_initialized(driver):
    """すでにDBに英語のデータが1件でも入っているか判定する"""
    query = "MATCH (u:Unit {subject: '英語'}) RETURN u LIMIT 1"
    with driver.session() as session:
        return session.run(query).single() is not None

def fetch_target_units(driver):
    """Neo4jから対象範囲のUnitノードを取得"""
    unit_map = {}
    with driver.session() as session:
        query = """
        MATCH (u:Unit)
        WHERE toInteger(u.unit_id) >= 220000001 AND toInteger(u.unit_id) <= 229999999
        RETURN u.unit_name, u.unit_id
        """
        for record in session.run(query):
            if record["u.unit_name"]:
                unit_map[record["u.unit_name"]] = record["u.unit_id"]
    return unit_map

def register_master_data(driver):
    """UnitとPropertyのマスタデータを登録（UNWINDで一括登録）"""
    query_unit = """
    UNWIND $batch AS data
    MERGE (u:Unit {unit_id: data.id})
    SET u.unit_name = data.name, u.subject = '英語', u.is_pre_defined = true, u.updated_at = datetime()
    """
    
    query_prop = """
    UNWIND $batch AS data
    MERGE (p:Property {property_id: data.id})
    SET p.property_name = data.name, p.subject = '英語', p.description = data.desc, p.is_pre_defined = true, p.updated_at = datetime()
    """
    
    unit_batch = [{"id": uid, "name": uname} for uid, uname in UNIT_DATA]
    prop_batch = [{"id": pid, "name": pname, "desc": "difficulty" if pid.startswith("021") else "part_of_speech"} for pid, pname in PROPERTY_DATA]

    with driver.session() as session:
        session.run(query_unit, batch=unit_batch)
        session.run(query_prop, batch=prop_batch)
    print("✅ English Master Data Registered (Upserted)")

def process_csv(driver):
    """CSVからConceptを登録して紐付け（UNWINDによる一括バルク処理で超高速化）"""
    if not os.path.exists(CSV_FILE):
        print(f"❌ File not found: {CSV_FILE}")
        return
        
    df = pd.read_csv(CSV_FILE, dtype=str)
    unit_map = fetch_target_units(driver)

    # 1. 一括登録用のリスト（メモリ上の箱）を準備
    concepts_batch = []
    unit_rels_batch = []
    prop_rels_batch = []

    # 2. Pythonの高速なループでリストを作る
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing English CSV Data"):
        c_id = row['id']
        if pd.isna(c_id): continue
        
        # Conceptデータ
        concepts_batch.append({"c_id": c_id, "c_name": row['Word']})
        
        # Unit紐付けデータ
        if pd.notna(row['Topic_raw']):
            for topic in [t.strip() for t in str(row['Topic_raw']).split(';')]:
                if topic in unit_map:
                    unit_rels_batch.append({"c_id": c_id, "u_id": unit_map[topic]})
        
        # Property紐付けデータ
        if pd.notna(row['Part of Speech']):
            prop_rels_batch.append({"c_id": c_id, "p_id": str(row['Part of Speech']).strip()})
        if pd.notna(row['Level']):
            prop_rels_batch.append({"c_id": c_id, "p_id": str(row['Level']).strip()})

    # 3. 準備したリストをNeo4jにドカンと1回ずつ送る
    with driver.session() as session:
        print("🚀 Registering Concepts to DB...")
        session.run("""
            UNWIND $batch AS data
            MERGE (c:Concept {concept_id: data.c_id})
            SET c.concept_name = data.c_name, c.is_pre_defined = true, c.updated_at = datetime()
        """, batch=concepts_batch)
        
        print("🚀 Linking Concepts to Units...")
        if unit_rels_batch:
            session.run("""
                UNWIND $batch AS data
                MATCH (c:Concept {concept_id: data.c_id}), (u:Unit {unit_id: data.u_id})
                MERGE (c)-[r:RELATED_TO]->(u)
                SET r.ratio = 1, r.rank = -1, r.updated_at = datetime()
            """, batch=unit_rels_batch)
            
        print("🚀 Linking Concepts to Properties...")
        if prop_rels_batch:
            session.run("""
                UNWIND $batch AS data
                MATCH (c:Concept {concept_id: data.c_id}), (p:Property {property_id: data.p_id})
                MERGE (c)-[r:BELONGS_TO]->(p)
                SET r.updated_at = datetime()
            """, batch=prop_rels_batch)

    print("✅ English CSV Processing Complete (Ultra-Fast)")

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    
    try:
        # DBの状態をチェックし、初期化済みならスキップ
        if check_already_initialized(driver):
            print("⚡ すでに英語の初期データは登録済みです。処理をすべてスキップします。")
            return
            
        register_master_data(driver)
        process_csv(driver)
        print("🎉 すべての英語ノードの初期登録が完了しました！")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    main()