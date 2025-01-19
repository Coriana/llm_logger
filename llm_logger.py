import sqlite3
import json
import logging
import threading
import queue
import uuid
from datetime import datetime
from typing import Dict, Any, Union

# Configure logging
logging.basicConfig(
    filename='llm_logger_errors.log',
    level=logging.ERROR,
    format='%(asctime)s %(levelname)s:%(message)s'
)

class LLMLogger:
    """
    A thread-safe logger for storing LLM interactions and token usage in an SQLite database using a logging queue.
    """

    def __init__(self, db_name: str = "llm_logs.db"):
        """
        Initializes the logger by setting up the queue and starting the logging thread.

        :param db_name: Name of the SQLite database file.
        """
        self.db_name = db_name
        self.log_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.logging_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.logging_thread.start()

    def _create_tables(self, conn):
        """
        Creates the interactions and token_usage tables if they don't already exist.
        """
        create_interactions_table = """
        CREATE TABLE IF NOT EXISTS interactions (
            interaction_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            model TEXT NOT NULL,
            request_data TEXT NOT NULL,
            response_data TEXT NOT NULL
        );
        """


        try:
            cursor = conn.cursor()
            cursor.execute(create_interactions_table)
            conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Failed to create tables: {e}")
            raise

    def _process_queue(self):
        """
        Processes log entries from the queue and writes them to the database.
        """
        try:
            conn = sqlite3.connect(
                self.db_name,
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
            )
            self._create_tables(conn)
        except sqlite3.Error as e:
            logging.error(f"Database connection failed: {e}")
            return

        while not self.stop_event.is_set() or not self.log_queue.empty():
            try:
                log_entry = self.log_queue.get(timeout=1)
            except queue.Empty:
                continue  # Check for stop_event again

            insert_interaction_query = """
            INSERT INTO interactions (interaction_id,
            timestamp,
            model,
            request_data,
            response_data,
            prompt_tokens)
            VALUES (?, ?, ?, ?, ?, ?);
            """

            try:
                cursor = conn.cursor()
                
                # Insert into interactions table
                cursor.execute(insert_interaction_query, (
                    log_entry['interaction_id'],
                    log_entry['timestamp'],
                    log_entry['model'],
                    log_entry['request_data'],
                    log_entry['response_data']
                ))

                conn.commit()
            except sqlite3.IntegrityError as e:
                logging.error(f"Integrity error logging interaction: {e}")
            except sqlite3.Error as e:
                logging.error(f"Database error logging interaction: {e}")
            except Exception as e:
                logging.error(f"Unexpected error logging interaction: {e}")
            finally:
                self.log_queue.task_done()

        conn.close()

    def _ensure_dict(self, data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """
        Ensures that the data is a dictionary. If it's a string, attempt to parse it as JSON.

        :param data: The data to ensure is a dictionary.
        :return: The data as a dictionary.
        :raises ValueError: If data is a string that cannot be parsed as JSON.
        """
        if isinstance(data, str):
            try:
                return json.loads(data)
            except json.JSONDecodeError as e:
                raise ValueError(f"String input is not valid JSON: {e}")
        elif isinstance(data, dict):
            return data
        else:
            raise TypeError("Data must be a dictionary or a JSON-formatted string.")

    def validate_interaction(self, request: Dict[str, Any], response: Dict[str, Any]) -> bool:
        """
        Validates the structure of request and response data.

        :param request: The request data as a dictionary.
        :param response: The response data as a dictionary.
        :return: True if valid, False otherwise.
        """
        required_request_keys = ["model", "messages"]
        required_response_keys = ["id", "object", "created", "model", "choices", "usage"]

        for key in required_request_keys:
            if key not in request:
                logging.error(f"Missing key in request: {key}")
                return False

        for key in required_response_keys:
            if key not in response:
                logging.error(f"Missing key in response: {key}")
                return False

        return True

    def log(
        self,
        request: Union[Dict[str, Any], str],
        response: Union[Dict[str, Any], str]
    ):
        """
        Logs a single interaction of request and response by enqueueing it.

        :param request: The request data as a dictionary or JSON string.
        :param response: The response data as a dictionary or JSON string.
        """
        try:
            request_dict = self._ensure_dict(request)
            response_dict = self._ensure_dict(response)
        except (ValueError, TypeError) as e:
            logging.error(f"Error processing input data: {e}")
            return

        # Validate interaction
        if not self.validate_interaction(request_dict, response_dict):
            logging.error("Invalid interaction data. Skipping logging.")
            return

        # Generate a unique interaction ID
        interaction_id = response_dict.get("id") or str(uuid.uuid4())

        # Extract model name from response
        model = response_dict.get("model")
        if not model:
            logging.error("Model name not found in response. Skipping logging.")
            return

        timestamp = datetime.utcnow().isoformat()

        # Serialize request and response data to JSON
        try:
            request_json = json.dumps(request_dict)
            response_json = json.dumps(response_dict)
        except (TypeError, OverflowError) as e:
            logging.error(f"Error serializing data to JSON: {e}")
            return

        # Extract token usage details
        usage = response_dict.get("usage", {})

        # Prepare the log entry
        log_entry = {
            'interaction_id': interaction_id,
            'timestamp': timestamp,
            'model': model,
            'request_data': request_json,
            'response_data': response_json,
            'token_usage': usage
        }

        # Enqueue the log entry
        self.log_queue.put(log_entry)

    def close(self):
        """
        Signals the logging thread to terminate and waits for it to finish.
        """
        self.stop_event.set()
        self.logging_thread.join()

    def __del__(self):
        """
        Ensures the logging thread is properly terminated when the instance is destroyed.
        """
        try:
            self.close()
        except:
            pass