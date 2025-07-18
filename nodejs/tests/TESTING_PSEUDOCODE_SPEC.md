# Node.js RAG Testing Implementation Pseudocode

## Module 1: Core Testing Infrastructure

### File: tests/conftest.js (< 500 lines)

```pseudocode
IMPORT environment_config, mock_factories, test_utilities

CLASS TestConfiguration:
    FUNCTION initialize():
        SET test_environment = ENVIRONMENT.NODE_ENV || 'test'
        SET iris_config = LOAD_IRIS_CONFIG_FROM_ENV()
        SET embedding_config = LOAD_EMBEDDING_CONFIG_FROM_ENV()
        SET performance_config = LOAD_PERFORMANCE_CONFIG_FROM_ENV()
        
        VALIDATE_CONFIGURATION(iris_config, embedding_config)
        SETUP_GLOBAL_TEST_VARIABLES()
        REGISTER_CLEANUP_HANDLERS()

FUNCTION load_iris_config_from_env():
    RETURN {
        host: ENVIRONMENT.TEST_IRIS_HOST || 'localhost',
        port: PARSE_INT(ENVIRONMENT.TEST_IRIS_PORT) || 1972,
        namespace: ENVIRONMENT.TEST_IRIS_NAMESPACE || 'USER',
        username: ENVIRONMENT.TEST_IRIS_USERNAME || 'SuperUser',
        password: ENVIRONMENT.TEST_IRIS_PASSWORD || 'SYS',
        use_real: ENVIRONMENT.USE_REAL_IRIS === 'true'
    }

FUNCTION create_mock_iris_connection():
    mock_connection = NEW MockIrisConnection()
    
    // Setup default mock responses
    mock_connection.set_mock_result('vector_search', [
        ['doc_001', 'Sample content 1', 'test.pdf', 1, 0, 0.95],
        ['doc_002', 'Sample content 2', 'test.pdf', 1, 1, 0.87]
    ])
    
    mock_connection.set_mock_result('insert_document', {
        rows_affected: 1,
        success: true
    })
    
    RETURN mock_connection

FUNCTION create_real_iris_connection():
    IF NOT TEST_CONFIG.iris.use_real:
        THROW ERROR('Real IRIS connection not enabled in test config')
    
    connection_manager = NEW IrisConnectionManager(TEST_CONFIG.iris)
    TRY:
        connection = AWAIT connection_manager.connect()
        RETURN connection
    CATCH error:
        LOG_WARNING('Failed to create real IRIS connection: ' + error.message)
        RETURN null

FUNCTION create_mock_embedding_utils():
    mock_utils = NEW MockEmbeddingUtils()
    mock_utils.set_embedding_dimension(384)
    mock_utils.set_deterministic_mode(true)
    RETURN mock_utils

FUNCTION create_real_embedding_utils():
    IF NOT TEST_CONFIG.embedding.use_real:
        THROW ERROR('Real embedding utils not enabled in test config')
    
    model_name = TEST_CONFIG.embedding.model || 'Xenova/all-MiniLM-L6-v2'
    utils = NEW EmbeddingUtils(model_name)
    AWAIT utils.initialize()
    RETURN utils

FUNCTION setup_test_database():
    connection = AWAIT create_test_connection()
    
    // Clean existing test data
    AWAIT connection.execute_query(
        "DELETE FROM SourceDocuments WHERE doc_id LIKE 'test_%'"
    )
    
    // Insert test fixtures
    FOR EACH document IN TEST_DOCUMENTS:
        AWAIT insert_test_document(connection, document)
    
    RETURN connection

FUNCTION cleanup_test_database(connection):
    TRY:
        AWAIT connection.execute_query(
            "DELETE FROM SourceDocuments WHERE doc_id LIKE 'test_%'"
        )
        AWAIT connection.close()
    CATCH error:
        LOG_WARNING('Cleanup failed: ' + error.message)

// Global test fixtures
EXPORT mock_iris_connection = create_mock_iris_connection
EXPORT real_iris_connection = create_real_iris_connection
EXPORT mock_embedding_utils = create_mock_embedding_utils
EXPORT real_embedding_utils = create_real_embedding_utils
EXPORT setup_test_database = setup_test_database
EXPORT cleanup_test_database = cleanup_test_database
```

### File: tests/jest.setup.js (< 500 lines)

```pseudocode
IMPORT test_config, performance_utils, assertion_helpers

FUNCTION setup_global_test_environment():
    // Setup global variables
    global.TEST_CONFIG = LOAD_TEST_CONFIGURATION()
    global.TEST_START_TIME = Date.now()
    
    // Setup performance monitoring
    global.measure_performance = CREATE_PERFORMANCE_MEASURER()
    global.expect_async = CREATE_ASYNC_EXPECTATION_HELPER()
    global.with_test_database = CREATE_DATABASE_TEST_HELPER()
    
    // Setup cleanup tracking
    global.cleanup_tasks = []
    global.register_cleanup = REGISTER_CLEANUP_TASK
    
    SETUP_GLOBAL_ERROR_HANDLERS()
    SETUP_UNHANDLED_REJECTION_HANDLERS()

FUNCTION create_performance_measurer():
    RETURN ASYNC FUNCTION measure_performance(operation_fn, label):
        start_time = performance.now()
        start_memory = process.memoryUsage()
        
        TRY:
            result = AWAIT operation_fn()
            end_time = performance.now()
            end_memory = process.memoryUsage()
            
            duration = end_time - start_time
            memory_delta = end_memory.heapUsed - start_memory.heapUsed
            
            LOG_PERFORMANCE_METRIC(label, duration, memory_delta)
            
            IF duration > TEST_CONFIG.performance.threshold_ms:
                LOG_WARNING('Performance threshold exceeded: ' + label)
            
            RETURN { result, duration, memory_delta }
        CATCH error:
            LOG_ERROR('Performance measurement failed: ' + error.message)
            THROW error

FUNCTION create_async_expectation_helper():
    RETURN ASYNC FUNCTION expect_async(promise_or_fn):
        TRY:
            IF IS_FUNCTION(promise_or_fn):
                result = AWAIT promise_or_fn()
            ELSE:
                result = AWAIT promise_or_fn
            
            RETURN expect(result)
        CATCH error:
            RETURN expect(error)

FUNCTION create_database_test_helper():
    RETURN ASYNC FUNCTION with_test_database(test_fn):
        connection = null
        TRY:
            connection = AWAIT create_test_connection()
            AWAIT setup_test_schema(connection)
            AWAIT test_fn(connection)
        FINALLY:
            IF connection:
                AWAIT cleanup_test_data(connection)
                AWAIT connection.close()

FUNCTION register_cleanup_task(cleanup_fn):
    global.cleanup_tasks.push(cleanup_fn)

FUNCTION setup_global_error_handlers():
    process.on('uncaughtException', FUNCTION(error):
        LOG_ERROR('Uncaught exception in test: ' + error.message)
        PERFORM_EMERGENCY_CLEANUP()
        process.exit(1)
    )
    
    process.on('unhandledRejection', FUNCTION(reason, promise):
        LOG_ERROR('Unhandled rejection in test: ' + reason)
        PERFORM_EMERGENCY_CLEANUP()
    )

// Jest lifecycle hooks
beforeAll(ASYNC FUNCTION():
    AWAIT setup_global_test_environment()
)

afterAll(ASYNC FUNCTION():
    AWAIT PERFORM_GLOBAL_CLEANUP()
)

beforeEach(FUNCTION():
    // Reset mocks and clear state
    jest.clearAllMocks()
    RESET_PERFORMANCE_COUNTERS()
)

afterEach(ASYNC FUNCTION():
    // Run registered cleanup tasks
    FOR EACH cleanup_task IN global.cleanup_tasks:
        TRY:
            AWAIT cleanup_task()
        CATCH error:
            LOG_WARNING('Cleanup task failed: ' + error.message)
    
    global.cleanup_tasks = []
)
```

## Module 2: Mock Implementations

### File: tests/mocks/iris_connector.js (< 500 lines)

```pseudocode
CLASS MockIrisConnection:
    CONSTRUCTOR(config = {}):
        this.config = config
        this.is_connected = false
        this.query_history = []
        this.mock_results = NEW Map()
        this.error_injection = NEW Map()
        this.call_count = 0

    ASYNC FUNCTION connect():
        IF this.error_injection.has('connect'):
            THROW this.error_injection.get('connect')
        
        this.is_connected = true
        this.call_count++
        RETURN this

    ASYNC FUNCTION execute_query(sql, params = []):
        this.call_count++
        query_record = {
            sql: sql,
            params: params,
            timestamp: Date.now(),
            call_number: this.call_count
        }
        this.query_history.push(query_record)
        
        // Check for error injection
        error_key = EXTRACT_QUERY_TYPE(sql)
        IF this.error_injection.has(error_key):
            THROW this.error_injection.get(error_key)
        
        // Return mock results based on query pattern
        result = MATCH_QUERY_PATTERN(sql, params)
        RETURN result

    FUNCTION match_query_pattern(sql, params):
        normalized_sql = sql.toLowerCase().trim()
        
        IF normalized_sql.includes('vector_cosine'):
            RETURN this.mock_results.get('vector_search') || DEFAULT_VECTOR_SEARCH_RESULTS()
        
        IF normalized_sql.includes('insert into sourcedocuments'):
            RETURN this.mock_results.get('insert_document') || DEFAULT_INSERT_RESULT()
        
        IF normalized_sql.includes('update sourcedocuments'):
            RETURN this.mock_results.get('update_document') || DEFAULT_UPDATE_RESULT()
        
        IF normalized_sql.includes('delete from sourcedocuments'):
            RETURN this.mock_results.get('delete_document') || DEFAULT_DELETE_RESULT()
        
        IF normalized_sql.includes('select count(*)'):
            RETURN this.mock_results.get('count_query') || [[42]]
        
        RETURN this.mock_results.get('default') || []

    FUNCTION set_mock_result(key, result):
        this.mock_results.set(key, result)

    FUNCTION inject_error(operation, error):
        this.error_injection.set(operation, error)

    FUNCTION get_query_history():
        RETURN this.query_history

    FUNCTION get_call_count():
        RETURN this.call_count

    FUNCTION reset():
        this.query_history = []
        this.call_count = 0
        this.mock_results.clear()
        this.error_injection.clear()

    ASYNC FUNCTION close():
        this.is_connected = false

FUNCTION default_vector_search_results():
    RETURN [
        ['test_doc_001', 'Sample content about vector search', 'test.pdf', 1, 0, 0.95],
        ['test_doc_002', 'Another sample about embeddings', 'test.pdf', 1, 1, 0.87],
        ['test_doc_003', 'Third sample about IRIS database', 'test.pdf', 2, 0, 0.82]
    ]

FUNCTION default_insert_result():
    RETURN { rows_affected: 1, success: true }

FUNCTION default_update_result():
    RETURN { rows_affected: 1, success: true }

FUNCTION default_delete_result():
    RETURN { rows_affected: 1, success: true }

FUNCTION extract_query_type(sql):
    normalized = sql.toLowerCase().trim()
    
    IF normalized.startswith('select'):
        IF normalized.includes('vector_cosine'):
            RETURN 'vector_search'
        RETURN 'select'
    
    IF normalized.startswith('insert'):
        RETURN 'insert'
    
    IF normalized.startswith('update'):
        RETURN 'update'
    
    IF normalized.startswith('delete'):
        RETURN 'delete'
    
    RETURN 'unknown'

CLASS MockIrisCursor:
    CONSTRUCTOR(connection):
        this.connection = connection
        this.current_result = null
        this.result_index = 0

    ASYNC FUNCTION execute(sql, params = []):
        this.current_result = AWAIT this.connection.execute_query(sql, params)
        this.result_index = 0

    FUNCTION fetchone():
        IF NOT this.current_result OR this.result_index >= this.current_result.length:
            RETURN null
        
        result = this.current_result[this.result_index]
        this.result_index++
        RETURN result

    FUNCTION fetchall():
        IF NOT this.current_result:
            RETURN []
        
        remaining = this.current_result.slice(this.result_index)
        this.result_index = this.current_result.length
        RETURN remaining

    FUNCTION close():
        this.current_result = null
        this.result_index = 0

EXPORT MockIrisConnection, MockIrisCursor
```

### File: tests/mocks/embedding_models.js (< 500 lines)

```pseudocode
CLASS MockEmbeddingUtils:
    CONSTRUCTOR(model_name = 'mock-model'):
        this.model_name = model_name
        this.is_initialized = false
        this.embedding_dimension = 384
        this.call_history = []
        this.deterministic_mode = true
        this.error_injection = NEW Map()
        this.custom_embeddings = NEW Map()

    ASYNC FUNCTION initialize():
        IF this.error_injection.has('initialize'):
            THROW this.error_injection.get('initialize')
        
        this.is_initialized = true
        this.call_history.push({
            operation: 'initialize',
            timestamp: Date.now()
        })

    ASYNC FUNCTION generate_embedding(text):
        IF NOT this.is_initialized:
            AWAIT this.initialize()
        
        IF this.error_injection.has('generate_embedding'):
            THROW this.error_injection.get('generate_embedding')
        
        this.call_history.push({
            operation: 'generate_embedding',
            text: text,
            timestamp: Date.now()
        })
        
        // Check for custom embedding
        IF this.custom_embeddings.has(text):
            RETURN this.custom_embeddings.get(text)
        
        // Generate deterministic or random embedding
        IF this.deterministic_mode:
            RETURN GENERATE_DETERMINISTIC_EMBEDDING(text, this.embedding_dimension)
        ELSE:
            RETURN GENERATE_RANDOM_EMBEDDING(this.embedding_dimension)

    ASYNC FUNCTION generate_batch_embeddings(texts, batch_size = 10):
        IF NOT this.is_initialized:
            AWAIT this.initialize()
        
        IF this.error_injection.has('generate_batch_embeddings'):
            THROW this.error_injection.get('generate_batch_embeddings')
        
        this.call_history.push({
            operation: 'generate_batch_embeddings',
            text_count: texts.length,
            batch_size: batch_size,
            timestamp: Date.now()
        })
        
        embeddings = []
        FOR EACH text IN texts:
            embedding = AWAIT this.generate_embedding(text)
            embeddings.push(embedding)
        
        RETURN embeddings

    FUNCTION generate_deterministic_embedding(text, dimension):
        hash = SIMPLE_HASH(text)
        embedding = []
        
        FOR i FROM 0 TO dimension - 1:
            // Generate deterministic values based on text hash and position
            value = Math.sin(hash + i) * 0.1
            embedding.push(value)
        
        // Normalize to unit vector
        RETURN NORMALIZE_VECTOR(embedding)

    FUNCTION generate_random_embedding(dimension):
        embedding = []
        FOR i FROM 0 TO dimension - 1:
            embedding.push((Math.random() - 0.5) * 0.2)
        
        RETURN NORMALIZE_VECTOR(embedding)

    FUNCTION normalize_vector(vector):
        magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0))
        IF magnitude === 0:
            RETURN vector
        
        RETURN vector.map(val => val / magnitude)

    FUNCTION simple_hash(str):
        hash = 0
        FOR i FROM 0 TO str.length - 1:
            char = str.charCodeAt(i)
            hash = ((hash << 5) - hash) + char
            hash = hash & hash  // Convert to 32-bit integer
        RETURN hash

    STATIC FUNCTION calculate_cosine_similarity(vector_a, vector_b):
        IF vector_a.length !== vector_b.length:
            THROW ERROR('Vectors must have same length')
        
        dot_product = 0
        norm_a = 0
        norm_b = 0
        
        FOR i FROM 0 TO vector_a.length - 1:
            dot_product += vector_a[i] * vector_b[i]
            norm_a += vector_a[i] * vector_a[i]
            norm_b += vector_b[i] * vector_b[i]
        
        norm_a = Math.sqrt(norm_a)
        norm_b = Math.sqrt(norm_b)
        
        IF norm_a === 0 OR norm_b === 0:
            RETURN 0
        
        RETURN dot_product / (norm_a * norm_b)

    STATIC FUNCTION preprocess_text(text, options = {}):
        max_length = options.max_length || 512
        remove_extra_whitespace = options.remove_extra_whitespace !== false
        to_lower_case = options.to_lower_case || false
        
        IF NOT text OR typeof text !== 'string':
            RETURN ''
        
        processed = text
        
        IF remove_extra_whitespace:
            processed = processed.replace(/\s+/g, ' ').trim()
        
        IF to_lower_case:
            processed = processed.toLowerCase()
        
        IF processed.length > max_length:
            processed = processed.substring(0, max_length).trim()
        
        RETURN processed

    STATIC FUNCTION chunk_text(text, options = {}):
        chunk_size = options.chunk_size || 500
        overlap = options.overlap || 50
        split_on_sentences = options.split_on_sentences !== false
        
        IF NOT text OR typeof text !== 'string':
            RETURN []
        
        chunks = []
        
        IF split_on_sentences:
            sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0)
            current_chunk = ''
            
            FOR EACH sentence IN sentences:
                trimmed = sentence.trim()
                IF current_chunk.length + trimmed.length <= chunk_size:
                    current_chunk += (current_chunk ? '. ' : '') + trimmed
                ELSE:
                    IF current_chunk:
                        chunks.push(current_chunk + '.')
                    current_chunk = trimmed
            
            IF current_chunk:
                chunks.push(current_chunk + '.')
        ELSE:
            FOR i FROM 0 TO text.length STEP (chunk_size - overlap):
                chunk = text.substring(i, i + chunk_size)
                chunks.push(chunk)
        
        RETURN chunks.filter(chunk => chunk.trim().length > 0)

    // Test utilities
    FUNCTION set_custom_embedding(text, embedding):
        this.custom_embeddings.set(text, embedding)

    FUNCTION inject_error(operation, error):
        this.error_injection.set(operation, error)

    FUNCTION set_deterministic_mode(enabled):
        this.deterministic_mode = enabled

    FUNCTION get_call_history():
        RETURN this.call_history

    FUNCTION reset():
        this.call_history = []
        this.custom_embeddings.clear()
        this.error_injection.clear()
        this.is_initialized = false

    FUNCTION get_embedding_dimension():
        RETURN this.embedding_dimension

    FUNCTION is_ready():
        RETURN this.is_initialized

    FUNCTION get_model_info():
        RETURN {
            model_name: this.model_name,
            is_initialized: this.is_initialized,
            embedding_dimension: this.embedding_dimension
        }

EXPORT MockEmbeddingUtils
```

## Module 3: Test Utilities

### File: tests/utils/test_helpers.js (< 500 lines)

```pseudocode
CLASS TestHelpers:
    STATIC ASYNC FUNCTION wait_for(condition_fn, timeout_ms = 5000):
        start_time = Date.now()
        
        WHILE Date.now() - start_time < timeout_ms:
            TRY:
                IF AWAIT condition_fn():
                    RETURN true
            CATCH error:
                // Continue waiting on errors
                pass
            
            AWAIT SLEEP(100)  // Wait 100ms before next check
        
        THROW ERROR('Condition not met within ' + timeout_ms + 'ms')

    STATIC FUNCTION generate_random_vector(dimension = 384):
        vector = []
        FOR i FROM 0 TO dimension - 1:
            vector.push(Math.random() - 0.5)
        
        RETURN NORMALIZE_VECTOR(vector)

    STATIC FUNCTION generate_test_document(id_suffix = ''):
        RETURN {
            doc_id: 'test_doc_' + id_suffix + '_' + Date.now(),
            title: 'Test Document ' + id_suffix,
            content: 'This is test content for document ' + id_suffix + '. It contains sample text for vector search testing.',
            source_file: 'test_' + id_suffix + '.pdf',
            page_number: 1,
            chunk_index: 0,
            embedding: GENERATE_RANDOM_VECTOR(384)
        }

    STATIC FUNCTION generate_test_documents(count):
        documents = []
        FOR i FROM 0 TO count - 1:
            documents.push(GENERATE_TEST_DOCUMENT(i.toString()))
        
        RETURN documents

    STATIC ASYNC FUNCTION cleanup_test_data(connection, test_prefix = 'test_'):
        TRY:
            AWAIT connection.execute_query(
                'DELETE FROM SourceDocuments WHERE doc_id LIKE ?',
                [test_prefix + '%']
            )
        CATCH error:
            LOG_WARNING('Test data cleanup failed: ' + error.message)

    STATIC FUNCTION assert_vector_similarity(vector1, vector2, threshold = 0.9):
        similarity = CALCULATE_COSINE_SIMILARITY(vector1, vector2)
        expect(similarity).toBeGreaterThan(threshold)
        RETURN similarity

    STATIC FUNCTION calculate_cosine_similarity(vector_a, vector_b):
        IF vector_a.length !== vector_b.length:
            THROW ERROR('Vectors must have same length')
        
        dot_product = 0
        norm_a = 0
        norm_b = 0
        
        FOR i FROM 0 TO vector_a.length - 1:
            dot_product += vector_a[i] * vector_b[i]
            norm_a += vector_a[i] * vector_a[i]
            norm_b += vector_b[i] * vector_b[i]
        
        norm_a = Math.sqrt(norm_a)
        norm_b = Math.sqrt(norm_b)
        
        IF norm_a === 0 OR norm_b === 0:
            RETURN 0
        
        RETURN dot_product / (norm_a * norm_b)

    STATIC FUNCTION normalize_vector(vector):
        magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0))
        IF magnitude === 0:
            RETURN vector
        
        RETURN vector.map(val => val / magnitude)

    STATIC FUNCTION assert_query_contains(query_history, expected_sql_pattern):
        found = query_history.some(query => 
            query.sql.toLowerCase().includes(expected_sql_pattern.toLowerCase())
        )
        expect(found).toBe(true)

    STATIC FUNCTION assert_query_parameters(query_history, expected_params):
        last_query = query_history[query_history.length - 1]
        expect(last_query.params).toEqual(expected_params)

    STATIC FUNCTION create_mock_pdf_content():
        RETURN {
            text: 'This is sample PDF content for testing. It contains multiple sentences. Each sentence provides test data for chunking and embedding.',
            pages: [
                {
                    page_number: 1,
                    text: 'First page content with vector search information.'
                },
                {
                    page_number: 2,
                    text: 'Second page content with database operations details.'
                }
            ]
        }

    STATIC ASYNC FUNCTION verify_database_state(connection, expected_doc_count):
        result = AWAIT connection.execute_query(
            'SELECT COUNT(*) FROM SourceDocuments WHERE doc_id LIKE ?',
            ['test_%']
        )
        
        actual_count = result[0][0]
        expect(actual_count).toBe(expected_doc_count)

    STATIC FUNCTION create_performance_benchmark():
        RETURN {
            start_time: performance.now(),
            start_memory: process.memoryUsage(),
            
            measure: FUNCTION():
                end_time = performance.now()
                end_memory = process.memoryUsage()
                
                RETURN {
                    duration: end_time - this.start_time,
                    memory_delta: end_memory.heapUsed - this.start_memory.heapUsed
                }
        }

    STATIC FUNCTION assert_performance_threshold(duration_ms, threshold_ms):
        expect(duration_ms).toBeLessThan(threshold_ms)
        
        IF duration_ms > threshold_ms * 0.8:
            console.warn('Performance close to threshold: ' + duration_ms + 'ms')

    STATIC ASYNC FUNCTION retry_operation(operation_fn, max_retries = 3, delay_ms = 1000):
        FOR attempt FROM 1 TO max_retries:
            TRY:
                RETURN AWAIT operation_fn()
            CATCH error:
                IF attempt === max_retries:
                    THROW error
                
                console.warn('Retry attempt ' + attempt + ' failed: ' + error.message)
                AWAIT SLEEP(delay_ms)

    STATIC FUNCTION sleep(ms):
        RETURN NEW Promise(resolve => setTimeout(resolve, ms))

EXPORT TestHelpers
```

This pseudocode specification provides the foundation for implementing a comprehensive testing framework that:

1. **Follows TDD principles** with clear test structure and mock strategies
2. **Aligns with rag-templates patterns** using similar fixture and configuration approaches  
3. **Provides comprehensive coverage** for unit, integration, and E2E testing
4. **Supports performance testing** with benchmarking and monitoring utilities
5. **Enables environment-based testing** with mock/real service switching
6. **Maintains modularity** with each module under 500 lines
7. **Includes error handling** and cleanup mechanisms
8. **Supports debugging** with detailed logging and state tracking

The implementation can proceed module by module, following the red-green-refactor TDD cycle with these specifications as the blueprint.