import os
from neo4j import GraphDatabase
from datetime import datetime

# ==========================================
# 設定
# ==========================================
NEO4J_URI = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))

# 学年データ
GRADE_DATA = [
    ("000000001", "小学1年"),
    ("000000002", "小学2年"),
    ("000000003", "小学3年"),
    ("000000004", "小学4年"),
    ("000000005", "小学5年"),
    ("000000006", "小学6年"),
    ("000000007", "中学1年"),
    ("000000008", "中学2年"),
    ("000000009", "中学3年"),
    ("000000010", "高校1年"),
    ("000000011", "高校2年"),
    ("000000012", "高校3年")
]

def register_grades(driver):
    """学年 (Property) ノードの登録（MERGEによる重複回避・上書き）"""
    
    query = """
    MERGE (p:Property {property_id: $id})
    SET p.property_name = $name,
        p.subject = "",
        p.description = "grades",
        p.updated_at = datetime(),
        p.is_pre_defined = true
    """
    
    print("🚀 学年ノードの登録を開始します...")
    with driver.session() as session:
        for gid, gname in GRADE_DATA:
            session.run(query, id=gid, name=gname)
            print(f"  - 登録完了: {gid} -> {gname}")
            
    print("✅ 学年ノードの登録が完了しました！")

def main():
    # データベースへ接続
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    
    try:
        register_grades(driver)
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    main()