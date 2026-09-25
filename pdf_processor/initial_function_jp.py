import os
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm
from datetime import datetime

# ==========================================
# 設定
# ==========================================
CSV_FILE = "jp_letters.csv"
NEO4J_URI = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))

# マスタデータ定義
UNIT_DATA_KOKUGO = [
    ("203000001", "ひらがな"), ("203000002", "カタカナ"), ("203000003", "漢字"),
    ("303000001", "清音"), ("303000002", "拗音"), ("303000003", "促音"), ("303000004", "撥音"),
    ("303000005", "長音"), ("303000006", "濁音"), ("303000007", "半濁音"), ("303000008", "拗濁音"), ("303000009", "拗半濁音"), ("303000010", "清音"), ("303000011", "拗音"), ("303000012", "促音"), ("303000013", "撥音"),
    ("303000014", "長音"), ("303000015", "濁音"), ("303000016", "半濁音"), ("303000017", "拗濁音"), ("303000018", "拗半濁音")
]
PART_OF_RELATIONS = [
    ("303000001", "203000001"), ("303000002", "203000001"), ("303000003", "203000001"),
    ("303000004", "203000001"), ("303000005", "203000001"), ("303000006", "203000001"),
    ("303000007", "203000001"), ("303000008", "203000001"), ("303000009", "203000001"),
    ("303000010", "203000002"), ("303000011", "203000002"), ("303000012", "203000002"),
    ("303000013", "203000002"), ("303000014", "203000002"), ("303000015", "203000002"),
    ("303000016", "203000002"), ("303000017", "203000002"), ("303000018", "203000002")
]
PROP_DIFFICULTY = [("031000001", "10級"), ("031000002", "9級"), ("031000003", "8級"), ("031000004", "7級"), ("031000005", "6級"), ("031000006", "5級"), ("031000007", "4級"), ("031000008", "3級"), ("031000009", "準2級"), ("031000010", "2級"), ("031000011", "準1級"), ("031000012", "1級"), ("031000013", "N5"), ("031000014", "N4"), ("031000015", "N3"), ("031000016", "N2"), ("031000017", "N1")]
PROP_RADICAL = [("030000001", "水"), ("030000002", "人"), ("030000003", "手"), ("030000004", "木"), ("030000005", "心"), ("030000006", "口"), ("030000007", "言"), ("030000008", "糸"), ("030000009", "辵"), ("030000010", "土"), ("030000011", "艸"), ("030000012", "肉"), ("030000013", "貝"), ("030000014", "宀"), ("030000015", "日"), ("030000016", "女"), ("030000017", "金"), ("030000018", "刀"), ("030000019", "阜"), ("030000020", "火"), ("030000021", "竹"), ("030000022", "力"), ("030000023", "禾"), ("030000024", "衣"), ("030000025", "頁"), ("030000026", "田"), ("030000027", "彳"), ("030000028", "攴"), ("030000029", "目"), ("030000030", "广"), ("030000031", "犬"), ("030000032", "大"), ("030000033", "山"), ("030000034", "巾"), ("030000035", "石"), ("030000036", "示"), ("030000037", "玉"), ("030000038", "疒"), ("030000039", "車"), ("030000040", "酉"), ("030000041", "尸"), ("030000042", "一"), ("030000043", "囗"), ("030000044", "雨"), ("030000045", "食"), ("030000046", "邑"), ("030000047", "十"), ("030000048", "寸"), ("030000049", "弓"), ("030000050", "足"), ("030000051", "馬"), ("030000052", "穴"), ("030000053", "米"), ("030000054", "虫"), ("030000055", "門"), ("030000056", "隹"), ("030000057", "儿"), ("030000058", "子"), ("030000059", "皿"), ("030000060", "八"), ("030000061", "曰"), ("030000062", "月"), ("030000063", "舟"), ("030000064", "冫"), ("030000065", "卩"), ("030000066", "又"), ("030000067", "戈"), ("030000068", "欠"), ("030000069", "止"), ("030000070", "网"), ("030000071", "羊"), ("030000072", "虍"), ("030000073", "見"), ("030000074", "乙"), ("030000075", "戶"), ("030000076", "方"), ("030000077", "殳"), ("030000078", "牛"), ("030000079", "耳"), ("030000080", "行"), ("030000081", "走"), ("030000082", "羽"), ("030000083", "二"), ("030000084", "亠"), ("030000085", "夕"), ("030000086", "工"), ("030000087", "干"), ("030000088", "彡"), ("030000089", "斤"), ("030000090", "歹"), ("030000091", "白"), ("030000092", "立"), ("030000093", "辛"), ("030000094", "丶"), ("030000095", "入"), ("030000096", "凵"), ("030000097", "厂"), ("030000098", "士"), ("030000099", "巛"), ("030000100", "幺"), ("030000101", "爪"), ("030000102", "矢"), ("030000103", "臼"), ("030000104", "襾"), ("030000105", "豕"), ("030000106", "里"), ("030000107", "骨"), ("030000108", "鬼"), ("030000109", "鳥"), ("030000110", "黑"), ("030000111", "丿"), ("030000112", "冂"), ("030000113", "冖"), ("030000114", "勹"), ("030000115", "匸"), ("030000116", "小"), ("030000117", "廴"), ("030000118", "斗"), ("030000119", "毋"), ("030000120", "老"), ("030000121", "至"), ("030000122", "舌"), ("030000123", "角"), ("030000124", "音"), ("030000125", "魚"), ("030000126", "鹿"), ("030000127", "丨"), ("030000128", "亅"), ("030000129", "匕"), ("030000130", "厶"), ("030000131", "廾"), ("030000132", "文"), ("030000133", "氏"), ("030000134", "片"), ("030000135", "玄"), ("030000136", "瓦"), ("030000137", "甘"), ("030000138", "生"), ("030000139", "疋"), ("030000140", "癶"), ("030000141", "缶"), ("030000142", "耒"), ("030000143", "臣"), ("030000144", "自"), ("030000145", "色"), ("030000146", "血"), ("030000147", "豆"), ("030000148", "赤"), ("030000149", "辰"), ("030000150", "釆"), ("030000151", "靑"), ("030000152", "革"), ("030000153", "麥"), ("030000154", "齊"), ("030000155", "齒"), ("030000156", "几"), ("030000157", "匚"), ("030000158", "卜"), ("030000159", "夊"), ("030000160", "尢"), ("030000161", "己"), ("030000162", "弋"), ("030000163", "彐"), ("030000164", "支"), ("030000165", "无"), ("030000166", "比"), ("030000167", "毛"), ("030000168", "气"), ("030000169", "父"), ("030000170", "爻"), ("030000171", "牙"), ("030000172", "瓜"), ("030000173", "用"), ("030000174", "皮"), ("030000175", "矛"), ("030000176", "而"), ("030000177", "聿"), ("030000178", "舛"), ("030000179", "艮"), ("030000180", "谷"), ("030000181", "豸"), ("030000182", "身"), ("030000183", "長"), ("030000184", "隶"), ("030000185", "非"), ("030000186", "面"), ("030000187", "韋"), ("030000188", "風"), ("030000189", "飛"), ("030000190", "首"), ("030000191", "香"), ("030000192", "高"), ("030000193", "髟"), ("030000194", "鬥"), ("030000195", "鬯"), ("030000196", "鹵"), ("030000197", "麻"), ("030000198", "黃"), ("030000199", "鼓"), ("030000200", "鼻"), ("030000201", "龍"), ("030000202", "龜"), ("030000203", "屮"), ("030000204", "ツ"), ("030000205", "戸"), ("030000206", "黒"), ("030000207", "歯"), ("030000208", "青")]
PROP_TOPIC = [("032000001", "numbers"), ("032000002", "days & times"), ("032000003", "people & school"), ("032000004", "transportation"), ("032000005", "nature & weather"), ("032000006", "adjective (color)"), ("032000007", "adjective (size)"), ("032000008", "adjective (shape)"), ("032000009", "verb"), ("032000010", "body & face"), ("032000011", "place"), ("032000012", "animal"), ("032000013", "adjective ()"), ("032000014", "adjective (height/price)"), ("032000015", "adjective (range)"), ("032000016", "adjective (age)"), ("032000017", "adjective (amount)"), ("032000018", "adjective (width)"), ("032000019", "people"), ("032000020", "seasons"), ("032000021", "food"), ("032000022", "adjective (price)"), ("032000023", "goods"), ("032000024", "concepts, society, study & abstract"), ("032000025", "position & direction"), ("032000026", "adjective (length)")]
PROP_PoS = [("033000001", "名詞"), ("033000002", "動詞"), ("033000003", "形容詞"),
("033000004", "副詞"), ("033000005", "連体詞"), ("033000006", "接続詞"), ("033000007", "助詞"),
("033000008", "助動詞"), ("033000009", "感動詞"), ("033000010", "記号"), ("033000011", "フィラー"), ("033000012", "接頭詞")]
DAKUON_PAIRS = [
    ("か", "が"), ("き", "ぎ"), ("く", "ぐ"), ("け", "げ"), ("こ", "ご"),
    ("さ", "ざ"), ("し", "じ"), ("す", "ず"), ("せ", "ぜ"), ("そ", "ぞ"),
    ("た", "だ"), ("ち", "ぢ"), ("つ", "づ"), ("て", "で"), ("と", "ど"),
    ("は", "ば"), ("ひ", "び"), ("ふ", "ぶ"), ("へ", "べ"), ("ほ", "ぼ"),
    ("カ", "ガ"), ("キ", "ギ"), ("ク", "グ"), ("ケ", "ゲ"), ("コ", "ゴ"),
    ("サ", "ザ"), ("シ", "ジ"), ("ス", "ズ"), ("セ", "ゼ"), ("ソ", "ゾ"),
    ("タ", "ダ"), ("チ", "ヂ"), ("ツ", "ヅ"), ("テ", "デ"), ("ト", "ド"),
    ("ハ", "バ"), ("ヒ", "ビ"), ("フ", "ブ"), ("ヘ", "ベ"), ("ホ", "ボ")
]
HANDAKUON_PAIRS = [
    ("は", "ぱ"), ("ひ", "ぴ"), ("ふ", "ぷ"), ("へ", "ぺ"), ("ほ", "ぽ"),
    ("ハ", "パ"), ("ヒ", "ピ"), ("フ", "プ"), ("ヘ", "ペ"), ("ホ", "ポ")
]

def format_yomi(val):
    if pd.isna(val): return ""
    val = str(val).strip()
    if val == "-": return ""
    return [v.strip() for v in val.split("・")] if "・" in val else val

def check_already_initialized(driver):
    """すでにDBに国語のデータが1件でも入っているか判定する"""
    query = "MATCH (u:Unit {subject: '国語'}) RETURN u LIMIT 1"
    with driver.session() as session:
        return session.run(query).single() is not None

def register_master_data(driver):
    """Unit, Unitの階層, PropertyのマスタデータをUNWINDで一括登録"""
    q_unit = """
    UNWIND $batch AS data
    MERGE (u:Unit {unit_id: data.id})
    SET u.unit_name = data.name, u.subject = '国語', u.is_pre_defined = true, u.updated_at = datetime()
    """
    
    q_part_of = """
    UNWIND $batch AS data
    MATCH (c:Unit {unit_id: data.cid}), (p:Unit {unit_id: data.pid})
    MERGE (c)-[r:PART_OF]->(p)
    SET r.updated_at = datetime()
    """
    
    q_prop = """
    UNWIND $batch AS data
    MERGE (p:Property {property_id: data.id})
    SET p.property_name = data.name, p.subject = '国語', p.description = data.desc, p.is_pre_defined = true, p.updated_at = datetime()
    """

    unit_batch = [{"id": uid, "name": uname} for uid, uname in UNIT_DATA_KOKUGO]
    part_of_batch = [{"cid": cid, "pid": pid} for cid, pid in PART_OF_RELATIONS]
    prop_batch = []
    
    for p_list, desc in [(PROP_DIFFICULTY, "difficulty"), (PROP_RADICAL, "radical"), (PROP_TOPIC, "トピック"), (PROP_PoS, "品詞")]:
        for pid, pname in p_list:
            prop_batch.append({"id": pid, "name": pname, "desc": desc})

    with driver.session() as session:
        session.run(q_unit, batch=unit_batch)
        session.run(q_part_of, batch=part_of_batch)
        if prop_batch:
            session.run(q_prop, batch=prop_batch)
            
    print("✅ Kokugo Master Data Registered (Upserted)")

def process_csv(driver):
    """CSVからConceptを登録して紐付け（UNWINDによる一括バルク処理で超高速化）"""
    if not os.path.exists(CSV_FILE):
        print(f"❌ File not found: {CSV_FILE}")
        return

    df = pd.read_csv(CSV_FILE, dtype=str)
    df.columns = df.columns.str.strip()

    concepts_batch = []
    unit_rels_batch = []
    prop_rels_batch = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing Kokugo CSV Data"):
        c_id = str(row.get('id')).strip()
        if not c_id or c_id == 'nan': continue

        num_strokes = str(row.get('num_strokes', '')).strip()
        if num_strokes.lower() == 'nan': num_strokes = ""

        c_name = str(row.get('name', '')).strip()
        yomi_on = format_yomi(row.get('attribute_on'))
        yomi_kun = format_yomi(row.get('attribute_kun'))

        concepts_batch.append({
            "c_id": c_id, "c_name": c_name, "num_strokes": num_strokes,
            "yomi_on": yomi_on, "yomi_kun": yomi_kun
        })

        # Unit紐付けデータ
        sub_u = str(row.get('subunit')).strip()
        main_u = str(row.get('unit')).strip()
        target_units = set()
        if main_u and main_u != 'nan': target_units.add(main_u)
        if sub_u and sub_u != 'nan': target_units.add(sub_u)
        
        for tu in target_units:
            unit_rels_batch.append({"c_id": c_id, "u_id": tu})

        # Property紐付けデータ
        for prop_col in ['radical', 'difficulty_1', 'difficulty_2', 'difficulty_3', 'topic']:
            prop_val = str(row.get(prop_col)).strip()
            if prop_val and prop_val != 'nan':
                for pid in prop_val.split(';'):
                    if pid.strip():
                        prop_rels_batch.append({"c_id": c_id, "p_id": pid.strip()})

    with driver.session() as session:
        print("🚀 Registering Kokugo Concepts to DB...")
        session.run("""
            UNWIND $batch AS data
            MERGE (c:Concept {concept_id: data.c_id})
            SET c.concept_name = data.c_name,
                c.num_strokes = data.num_strokes,
                c.yomi_on = data.yomi_on,
                c.yomi_kun = data.yomi_kun,
                c.subject = "国語",
                c.updated_at = datetime(),
                c.is_pre_defined = true
        """, batch=concepts_batch)
        
        print("🚀 Linking Concepts to Units...")
        if unit_rels_batch:
            session.run("""
                UNWIND $batch AS data
                MATCH (c:Concept {concept_id: data.c_id}), (u:Unit {unit_id: data.u_id})
                MERGE (c)-[r:BELONGS_TO]->(u)
                SET r.updated_at = datetime()
            """, batch=unit_rels_batch)
            
        print("🚀 Linking Concepts to Properties...")
        if prop_rels_batch:
            session.run("""
                UNWIND $batch AS data
                MATCH (c:Concept {concept_id: data.c_id}), (p:Property {property_id: data.p_id})
                MERGE (c)-[r:BELONGS_TO]->(p)
                SET r.updated_at = datetime()
            """, batch=prop_rels_batch)

    print("✅ Kokugo CSV Processing Complete (Ultra-Fast)")

def link_phonetic_variants(driver):
    """濁音・半濁音から清音へ FOLLOWS エッジを結ぶ（UNWINDで一括登録）"""
    q_link = """
    UNWIND $batch AS data
    MATCH (base:Concept {concept_name: data.base_char})
    MATCH (variant:Concept {concept_name: data.variant_char})
    MERGE (variant)-[r:FOLLOWS]->(base)
    SET r.description = data.desc, r.updated_at = datetime()
    """
    
    batch = []
    for base_char, variant_char in DAKUON_PAIRS:
        batch.append({"base_char": base_char, "variant_char": variant_char, "desc": "濁音"})
    for base_char, variant_char in HANDAKUON_PAIRS:
        batch.append({"base_char": base_char, "variant_char": variant_char, "desc": "半濁音"})
        
    print("🔗 Linking Phonetic Variants (Dakuon & Handakuon)...")
    with driver.session() as session:
        if batch:
            session.run(q_link, batch=batch)
    print("✅ Phonetic Variants Linked Successfully")

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    
    try:
        if check_already_initialized(driver):
            print("⚡ すでに国語の初期データは登録済みです。処理をすべてスキップします。")
            return
            
        register_master_data(driver)
        process_csv(driver)
        link_phonetic_variants(driver)
        print("🎉 すべての国語ノードの初期登録が完了しました！")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    main()