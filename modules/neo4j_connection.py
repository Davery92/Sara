"""
Module to check Neo4j database connection.
"""

from neo4j import GraphDatabase
import logging

# Configure logging
logger = logging.getLogger("neo4j-connection")

# Neo4j connection details
NEO4J_URI = "bolt://10.185.1.8:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Nutman17!"

def check_neo4j_connection():
    """
    Check the connection to the Neo4j database.
    Returns "Connected" if successful, otherwise returns an error message.
    """
    try:
        # Initialize the Neo4j driver
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        # Test the connection by running a simple query
        with driver.session() as session:
            result = session.run("RETURN 1 as num")
            record = result.single()

            # Check if the query returned the expected result
            if record and record["num"] == 1:
                logger.info("Neo4j connection successful!")
                return "Connected"
            else:
                logger.warning("Neo4j connection failed: Unexpected query result")
                return "Disconnected"

    except Exception as e:
        # Log the error and return an appropriate message
        logger.error(f"Error checking Neo4j connection: {e}")
        return f"Error: {str(e)}"

    finally:
        # Ensure the driver is properly closed
        if 'driver' in locals():
            driver.close()

# Example usage (for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    status = check_neo4j_connection()
    print(f"Neo4j connection status: {status}")