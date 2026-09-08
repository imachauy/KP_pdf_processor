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
    """UnitとPropertyのマスタデータを登録（MERGEにより重複回避・上書き）"""
    # 変更: u.is_pre_defined = true を追加
    query_unit = "MERGE (u:Unit {unit_id: $id}) SET u.unit_name = $name, u.subject = '英語', u.is_pre_defined = true, u.updated_at = datetime()"
    
    # 変更: p.is_pre_defined = true を追加
    query_prop = "MERGE (p:Property {property_id: $id}) SET p.property_name = $name, p.subject = '英語', p.description = $desc, p.is_pre_defined = true, p.updated_at = datetime()"
    
    with driver.session() as session:
        for uid, uname in UNIT_DATA:
            session.run(query_unit, id=uid, name=uname)
        for pid, pname in PROPERTY_DATA:
            desc = "difficulty" if pid.startswith("021") else "part_of_speech"
            session.run(query_prop, id=pid, name=pname, desc=desc)
    print("✅ English Master Data Registered (Upserted)")

def process_csv(driver):
    """CSVからConceptを登録して紐付け（すべてMERGEに変更し上書き対応）"""
    if not os.path.exists(CSV_FILE):
        print(f"❌ File not found: {CSV_FILE}")
        return
        
    df = pd.read_csv(CSV_FILE, dtype=str)
    unit_map = fetch_target_units(driver)

    # CREATEからMERGE + SETへ変更
    q_concept = """
    MERGE (c:Concept {concept_id: $c_id})
    SET c.concept_name = $c_name,
        c.is_pre_defined = true,
        c.updated_at = datetime()
    """
    
    # RELATED_TOエッジの重複防止とプロパティ上書き
    q_link_unit = """
    MATCH (c:Concept {concept_id: $c_id}), (u:Unit {unit_id: $u_id})
    MERGE (c)-[r:RELATED_TO]->(u)
    SET r.ratio = 1, r.rank = -1, r.updated_at = datetime()
    """
    
    # BELONGS_TOエッジの重複防止
    q_link_prop = """
    MATCH (c:Concept {concept_id: $c_id}), (p:Property {property_id: $p_id})
    MERGE (c)-[r:BELONGS_TO]->(p)
    SET r.updated_at = datetime()
    """

    with driver.session() as session:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing English CSV"):
            c_id = row['id']
            if pd.isna(c_id): continue
            
            # Concept作成・上書き
            session.run(q_concept, c_id=c_id, c_name=row['Word'])
            
            # Unit紐付け (エッジの作成・上書き)
            if pd.notna(row['Topic_raw']):
                for topic in [t.strip() for t in str(row['Topic_raw']).split(';')]:
                    if topic in unit_map:
                        session.run(q_link_unit, c_id=c_id, u_id=unit_map[topic])
            
            # Property紐付け (エッジの作成・上書き)
            if pd.notna(row['Part of Speech']):
                session.run(q_link_prop, c_id=c_id, p_id=str(row['Part of Speech']).strip())
            if pd.notna(row['Level']):
                session.run(q_link_prop, c_id=c_id, p_id=str(row['Level']).strip())
    print("✅ English CSV Processing Complete (Upserted)")

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    register_master_data(driver)
    process_csv(driver)
    driver.close()

if __name__ == "__main__":
    main()