# MCP Integration Guide

This guide provides detailed information on integrating with the Multi-Cloud Platform (MCP) and utilizing the IRIS SQL Tool within your RAG applications.

## 1. Overview of MCP Integration

The MCP integration allows you to deploy and manage RAG services as microservices, providing a flexible and scalable architecture. This design facilitates seamless integration with various enterprise systems and cloud environments.

## 2. Creating MCP Servers

To create an MCP server, use the `createMCPServer` function from the `@rag-templates/mcp` package.

### Example:
```javascript
import { createMCPServer } from '@rag-templates/mcp';

const server = createMCPServer({
    name: "my-rag-server",
    description: "A custom RAG server for specific knowledge base",
    ragConfig: {
        technique: 'basic', // or 'colbert', 'crag', etc.
        llm_provider: 'openai',
        embedding_model: 'text-embedding-ada-002'
    },
    // Additional server configurations can be added here
    port: 3000
});

server.start();
```

### Configuration Options for `createMCPServer`:

-   `name`: (String, required) A unique name for your MCP server.
-   `description`: (String, optional) A brief description of the server's purpose.
-   `ragConfig`: (Object, required) Configuration for the RAG pipeline. This object can include:
    -   `technique`: (String) The RAG technique to use (e.g., 'basic', 'colbert', 'crag', 'hyde', 'graphrag', 'hybrid_ifind', 'noderag').
    -   `llm_provider`: (String) The LLM provider (e.g., 'openai', 'azure_openai').
    -   `embedding_model`: (String) The embedding model to use.
    -   Any other valid RAG configuration parameters.
-   `port`: (Number, optional) The port on which the MCP server will listen. Defaults to a system-assigned port if not specified.

## 3. IRIS SQL Tool Integration

The IRIS SQL Tool provides direct SQL access and advanced vector search capabilities, allowing you to interact with your InterSystems IRIS database directly from your MCP-deployed RAG services. This is particularly useful for:

-   **Complex Data Retrieval**: Executing custom SQL queries to fetch data that might not be directly accessible via standard RAG pipeline methods.
-   **Vector Search**: Leveraging IRIS's native vector search capabilities for highly efficient similarity searches.
-   **Data Management**: Performing CRUD operations on your RAG data within the IRIS database.

### How to Use the IRIS SQL Tool:

The IRIS SQL Tool is exposed through the MCP server's API. You can send SQL queries or vector search requests to the MCP server, which will then execute them against the configured InterSystems IRIS database.

### Example (Conceptual API Call):

```javascript
// Assuming you have an MCP server running at http://localhost:3000
const mcpServerUrl = 'http://localhost:3000';

// Example: Execute a SQL query
async function executeSqlQuery(query) {
    const response = await fetch(`${mcpServerUrl}/sql`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: query })
    });
    return response.json();
}

// Example: Perform a vector search
async function performVectorSearch(vector, topK) {
    const response = await fetch(`${mcpServerUrl}/vector-search`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ vector: vector, topK: topK })
    });
    return response.json();
}

// Usage
// executeSqlQuery("SELECT TOP 10 * FROM MyDocuments WHERE Category = 'Science'")
//     .then(data => console.log("SQL Query Result:", data))
//     .catch(error => console.error("Error executing SQL query:", error));

// performVectorSearch([0.1, 0.2, ..., 0.N], 5) // Replace with actual vector
//     .then(data => console.log("Vector Search Result:", data))
//     .catch(error => console.error("Error performing vector search:", error));
```

**Note**: The exact API endpoints and request/response formats for the IRIS SQL Tool will depend on the implementation within the MCP server. Refer to the server's API documentation or source code for precise details.

## 4. Deployment and Management

Details on deploying MCP servers to various environments (e.g., Docker, Kubernetes) and managing their lifecycle will be covered in separate deployment guides.

## 5. Troubleshooting

For common issues and troubleshooting tips related to MCP integration and the IRIS SQL Tool, refer to the project's main documentation or open an issue on the GitHub repository.