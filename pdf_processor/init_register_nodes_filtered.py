import os
import pandas as pd
from neo4j import GraphDatabase
from datetime import datetime

# ==========================================
# 設定
# ==========================================
URI = os.getenv("NEO4J_URL", "bolt://localhost:7687")
AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
CSV_FILE = "node.csv"

# ==========================================
# ロジック定義
# ==========================================
def get_node_type(nid):
    """IDに基づいてノードの種類とIDキー、ラベルを判定する"""
    # 2億番台 ～ 9億8999万9999 -> Unit
    if 200000000 < nid < 990000000:
        return "Unit", "unit_id"
    # 9億9000万以上 -> Concept
    elif nid >= 990000000:
        return "Concept", "concept_id"
    # それ以外（1億番台など） -> 対象外
    else:
        return None, None

def main():
    print(f"📖 Reading {CSV_FILE}...")
    try:
        # CSVの読み込み（全てのカラムを文字列として読み込む）
        df = pd.read_csv(CSV_FILE, dtype=str)
    except FileNotFoundError:
        print(f"❌ File not found: {CSV_FILE}")
        return

    driver = GraphDatabase.driver(URI, auth=AUTH)
    current_time = datetime.utcnow().isoformat() + "Z"
    
    stats = {"Unit": 0, "Concept": 0, "Skipped": 0, "Relations": 0}

    print("🚀 Starting import...")

    with driver.session() as session:
        for index, row in df.iterrows():
            try:
                # 1. ノード情報の取得
                nid_str = row.get('node_id')
                if not nid_str: continue
                
                nid = int(nid_str)
                name = row.get('node', '')
                school_year = row.get('school_year', '')  # カラムがない場合は空文字
                part_of_id_str = row.get('part_of')       # 親ID

                # 2. 種類の判定 (2億番台:Unit, 9億番台:Concept)
                label, id_key = get_node_type(nid)

                # 登録対象外（1億番台など）はスキップ
                if not label:
                    stats["Skipped"] += 1
                    continue

                # 3. ノード登録クエリの構築
                # ラベルを動的に埋め込む (f-string)
                # 注意: 外部入力をそのままラベルに使う場合はインジェクションに注意が必要ですが、
                # ここでは get_node_type で固定値(Unit/Concept)しか返さないため安全です。
                query_create_node = f"""
                MERGE (n:{label} {{ {id_key}: $id }})
                SET n.{label.lower()}_name = $name
                """
                
                # school_yearがあればセットするクエリを追加
                if school_year and pd.notna(school_year):
                     query_create_node += f", n.school_year = $school_year"

                # Conceptの場合は作成日時なども入れたい場合（任意）
                if label == "Concept":
                    query_create_node += ", n.created_at = $created_at"

                # ノード作成実行
                session.run(query_create_node, 
                            id=nid_str, 
                            name=name, 
                            school_year=school_year,
                            created_at=current_time)
                stats[label] += 1

                # 4. PART_OF 関係の登録
                if part_of_id_str and pd.notna(part_of_id_str):
                    part_of_id = int(part_of_id_str)
                    parent_label, parent_id_key = get_node_type(part_of_id)

                    # 親も登録対象の範囲内（Unit/Concept）である場合のみ関係を作成
                    # ※1億番台（登録しない対象）へのリンクは、親ノードが存在しないため作成しません
                    if parent_label:
                        query_relation = f"""
                        MATCH (child:{label} {{ {id_key}: $child_id }})
                        MERGE (parent:{parent_label} {{ {parent_id_key}: $parent_id }})
                        MERGE (child)-[:PART_OF]->(parent)
                        """
                        session.run(query_relation, child_id=nid_str, parent_id=part_of_id_str)
                        stats["Relations"] += 1

            except Exception as e:
                print(f"⚠️ Error at row {index}: {e}")

    driver.close()

    print("-" * 30)
    print("✅ Import Completed!")
    print(f"   - Units Created   : {stats['Unit']}")
    print(f"   - Concepts Created: {stats['Concept']}")
    print(f"   - Skipped         : {stats['Skipped']}")
    print(f"   - PART_OF Relations: {stats['Relations']}")
    print("-" * 30)

if __name__ == "__main__":
    main()