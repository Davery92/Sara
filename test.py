from neo4j import GraphDatabase

def test_neo4j_connection():
    uri = "bolt://10.185.1.8:7687"
    user = "neo4j"
    password = "Nutman17!"
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        with driver.session() as session:
            result = session.run("RETURN 1 as num")
            record = result.single()
            if record and record["num"] == 1:
                print("Neo4j connection successful!")
            else:
                print("Neo4j connection failed!")
    except Exception as e:
        print(f"Neo4j connection error: {e}")
    finally:
        driver.close()

test_neo4j_connection()