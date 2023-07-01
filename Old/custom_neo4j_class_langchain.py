from typing import Any, Dict, List, Optional
import neo4j

class Neo4jGraph:
    """Neo4j wrapper for graph operations."""
    def __init__(
        self,
        driver: Optional[Any] = None,
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ) -> None:
        """Create a new Neo4j graph wrapper instance."""
        try:
            import neo4j
        except ImportError:
            raise ValueError(
                "Could not import neo4j python package. "
                "Please install it with `pip install neo4j`."
            )

        if driver is not None:
            self._driver = driver
        elif url is not None and username is not None and password is not None:
            self._driver = neo4j.GraphDatabase.driver(url, auth=(username, password))
        else:
            raise ValueError(
                "Either provide a driver instance or url, username, and password."
            )

        self._database = database or "neo4j"

        # Verify connection
        try:
            self._driver.verify_connectivity()
        except neo4j.exceptions.ServiceUnavailable:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the url is correct"
            )
        except neo4j.exceptions.AuthError:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the username and password are correct"
            )

    def close(self) -> None:
        """Close the Neo4j driver."""
        self._driver.close()

    def run_query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Query Neo4j database."""
        from neo4j.exceptions import CypherSyntaxError

        with self._driver.session(database=self._database) as session:
            try:
                data = session.run(query, params)
                return [r.data() for r in data]
            except CypherSyntaxError as e:
                raise ValueError("Generated Cypher Statement is not valid\n" f"{e}")



