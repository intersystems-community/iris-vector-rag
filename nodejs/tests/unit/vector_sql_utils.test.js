// rag-templates/nodejs/tests/unit/vector_sql_utils.test.js
// Unit tests for VectorSQLUtils following TDD and rag-templates patterns

const VectorSQLUtils = require('../../src/vector_sql_utils');

describe('VectorSQLUtils', () => {
  describe('validateVectorString', () => {
    test('should accept valid vector strings with various formats', () => {
      // Standard comma-separated format
      expect(VectorSQLUtils.validateVectorString('1.0,2.0,3.0')).toBe(true);
      expect(VectorSQLUtils.validateVectorString('0.1, 0.2, 0.3')).toBe(true);
      expect(VectorSQLUtils.validateVectorString('-1.5,2.7,-3.9')).toBe(true);
      
      // High precision numbers (common in embeddings)
      expect(VectorSQLUtils.validateVectorString('0.123456789,-0.987654321')).toBe(true);
      expect(VectorSQLUtils.validateVectorString('1e-5,2.5e-3,3.14159')).toBe(true);
      
      // Edge cases for valid vectors
      expect(VectorSQLUtils.validateVectorString('0')).toBe(true);
      expect(VectorSQLUtils.validateVectorString('0.0')).toBe(true);
      expect(VectorSQLUtils.validateVectorString('-0.0')).toBe(true);
    });

    test('should reject malicious SQL injection attempts', () => {
      // SQL injection patterns
      expect(VectorSQLUtils.validateVectorString("'; DROP TABLE SourceDocuments; --")).toBe(false);
      expect(VectorSQLUtils.validateVectorString("1.0'; DELETE FROM users; --")).toBe(false);
      expect(VectorSQLUtils.validateVectorString("UNION SELECT * FROM users")).toBe(false);
      expect(VectorSQLUtils.validateVectorString("1.0 OR 1=1")).toBe(false);
      
      // XSS and script injection
      expect(VectorSQLUtils.validateVectorString("<script>alert('xss')</script>")).toBe(false);
      expect(VectorSQLUtils.validateVectorString("javascript:alert(1)")).toBe(false);
      
      // Command injection
      expect(VectorSQLUtils.validateVectorString("1.0; rm -rf /")).toBe(false);
      expect(VectorSQLUtils.validateVectorString("$(cat /etc/passwd)")).toBe(false);
    });

    test('should reject invalid vector formats', () => {
      // Wrong separators
      expect(VectorSQLUtils.validateVectorString('1.0;2.0;3.0')).toBe(false);
      expect(VectorSQLUtils.validateVectorString('1.0|2.0|3.0')).toBe(false);
      expect(VectorSQLUtils.validateVectorString('1.0:2.0:3.0')).toBe(false);
      
      // Invalid characters
      expect(VectorSQLUtils.validateVectorString('1.0,abc,3.0')).toBe(false);
      expect(VectorSQLUtils.validateVectorString('1.0,2.0,3.0!')).toBe(false);
      expect(VectorSQLUtils.validateVectorString('1.0,2.0,3.0@')).toBe(false);
      
      // Malformed numbers
      expect(VectorSQLUtils.validateVectorString('1..0,2.0,3.0')).toBe(false);
      expect(VectorSQLUtils.validateVectorString('1.0,,3.0')).toBe(false);
      expect(VectorSQLUtils.validateVectorString(',1.0,2.0,3.0')).toBe(false);
      expect(VectorSQLUtils.validateVectorString('1.0,2.0,3.0,')).toBe(false);
    });

    test('should handle edge cases and null inputs', () => {
      expect(VectorSQLUtils.validateVectorString('')).toBe(false);
      expect(VectorSQLUtils.validateVectorString(null)).toBe(false);
      expect(VectorSQLUtils.validateVectorString(undefined)).toBe(false);
      expect(VectorSQLUtils.validateVectorString(123)).toBe(false);
      expect(VectorSQLUtils.validateVectorString([])).toBe(false);
      expect(VectorSQLUtils.validateVectorString({})).toBe(false);
    });
  });

  describe('validateTopK', () => {
    test('should accept valid topK values', () => {
      expect(VectorSQLUtils.validateTopK(1)).toBe(true);
      expect(VectorSQLUtils.validateTopK(10)).toBe(true);
      expect(VectorSQLUtils.validateTopK(100)).toBe(true);
      expect(VectorSQLUtils.validateTopK(1000)).toBe(true);
    });

    test('should reject invalid topK values', () => {
      expect(VectorSQLUtils.validateTopK(0)).toBe(false);
      expect(VectorSQLUtils.validateTopK(-1)).toBe(false);
      expect(VectorSQLUtils.validateTopK(1.5)).toBe(false);
      expect(VectorSQLUtils.validateTopK('10')).toBe(false);
      expect(VectorSQLUtils.validateTopK(null)).toBe(false);
      expect(VectorSQLUtils.validateTopK(undefined)).toBe(false);
      expect(VectorSQLUtils.validateTopK([])).toBe(false);
      expect(VectorSQLUtils.validateTopK({})).toBe(false);
    });

    test('should handle edge cases for large values', () => {
      expect(VectorSQLUtils.validateTopK(Number.MAX_SAFE_INTEGER)).toBe(true);
      expect(VectorSQLUtils.validateTopK(Infinity)).toBe(false);
      expect(VectorSQLUtils.validateTopK(-Infinity)).toBe(false);
      expect(VectorSQLUtils.validateTopK(NaN)).toBe(false);
    });
  });

  describe('formatVectorSearchSQL', () => {
    const validOptions = {
      tableName: 'SourceDocuments',
      vectorColumn: 'embedding',
      vectorString: '0.1,0.2,0.3',
      embeddingDim: 384,
      topK: 5
    };

    test('should generate valid SQL with minimal options', () => {
      const sql = VectorSQLUtils.formatVectorSearchSQL(validOptions);
      
      expect(sql).toContain('SELECT TOP 5');
      expect(sql).toContain('doc_id');
      expect(sql).toContain('text_content');
      expect(sql).toContain('source_file, page_number, chunk_index');
      expect(sql).toContain('VECTOR_COSINE(embedding, TO_VECTOR(\'0.1,0.2,0.3\', \'FLOAT\', 384))');
      expect(sql).toContain('FROM SourceDocuments');
      expect(sql).toContain('WHERE embedding IS NOT NULL');
      expect(sql).toContain('ORDER BY score DESC');
    });

    test('should include custom columns when specified', () => {
      const options = {
        ...validOptions,
        idColumn: 'custom_id',
        contentColumn: 'custom_content'
      };
      
      const sql = VectorSQLUtils.formatVectorSearchSQL(options);
      
      expect(sql).toContain('custom_id');
      expect(sql).toContain('custom_content');
      expect(sql).not.toContain('doc_id');
      expect(sql).not.toContain('text_content');
    });

    test('should include additional WHERE conditions', () => {
      const options = {
        ...validOptions,
        additionalWhere: "source_file = 'test.pdf'"
      };
      
      const sql = VectorSQLUtils.formatVectorSearchSQL(options);
      
      expect(sql).toContain("source_file = 'test.pdf'");
      expect(sql).toContain('AND');
    });

    test('should handle null content column', () => {
      const options = {
        ...validOptions,
        contentColumn: null
      };
      
      const sql = VectorSQLUtils.formatVectorSearchSQL(options);
      
      expect(sql).not.toContain('text_content');
      expect(sql).toContain('doc_id');
    });

    test('should validate and reject invalid table names', () => {
      const invalidOptions = {
        ...validOptions,
        tableName: 'invalid-table; DROP TABLE users;'
      };
      
      expect(() => {
        VectorSQLUtils.formatVectorSearchSQL(invalidOptions);
      }).toThrow('Invalid table name');
    });

    test('should validate and reject invalid vector strings', () => {
      const invalidOptions = {
        ...validOptions,
        vectorString: 'invalid; DROP TABLE users;'
      };
      
      expect(() => {
        VectorSQLUtils.formatVectorSearchSQL(invalidOptions);
      }).toThrow('Invalid vector string');
    });

    test('should validate and reject invalid topK values', () => {
      const invalidOptions = {
        ...validOptions,
        topK: -1
      };
      
      expect(() => {
        VectorSQLUtils.formatVectorSearchSQL(invalidOptions);
      }).toThrow('Invalid top_k value');
    });

    test('should handle large embedding dimensions', () => {
      const options = {
        ...validOptions,
        embeddingDim: 1536,
        vectorString: Array(1536).fill(0.1).join(',')
      };
      
      const sql = VectorSQLUtils.formatVectorSearchSQL(options);
      expect(sql).toContain('TO_VECTOR');
      expect(sql).toContain('1536');
    });
  });

  describe('formatVectorForIris', () => {
    test('should format vector arrays correctly', () => {
      const vector = [0.1, 0.2, 0.3, 0.4];
      const formatted = VectorSQLUtils.formatVectorForIris(vector);
      
      expect(formatted).toBe('0.1,0.2,0.3,0.4');
    });

    test('should handle negative numbers', () => {
      const vector = [-0.1, 0.2, -0.3, 0.4];
      const formatted = VectorSQLUtils.formatVectorForIris(vector);
      
      expect(formatted).toBe('-0.1,0.2,-0.3,0.4');
    });

    test('should handle empty arrays', () => {
      const vector = [];
      const formatted = VectorSQLUtils.formatVectorForIris(vector);
      
      expect(formatted).toBe('');
    });

    test('should handle single element arrays', () => {
      const vector = [0.5];
      const formatted = VectorSQLUtils.formatVectorForIris(vector);
      
      expect(formatted).toBe('0.5');
    });

    test('should handle high precision numbers', () => {
      const vector = [0.123456789, -0.987654321, 0.000000001];
      const formatted = VectorSQLUtils.formatVectorForIris(vector);
      
      expect(formatted).toBe('0.123456789,-0.987654321,0.000000001');
    });

    test('should handle large vectors efficiently', () => {
      const largeVector = Array(1536).fill(0).map((_, i) => i / 1536);
      const formatted = VectorSQLUtils.formatVectorForIris(largeVector);
      
      expect(formatted.split(',').length).toBe(1536);
      expect(formatted).toContain('0');
      expect(formatted).toContain('0.999');
    });
  });

  describe('SQL injection protection', () => {
    test('should prevent SQL injection in table name', () => {
      const maliciousOptions = {
        tableName: "users; DROP TABLE SourceDocuments; --",
        vectorColumn: 'embedding',
        vectorString: '0.1,0.2,0.3',
        embeddingDim: 384,
        topK: 5
      };
      
      expect(() => {
        VectorSQLUtils.formatVectorSearchSQL(maliciousOptions);
      }).toThrow('Invalid table name');
    });

    test('should prevent SQL injection in vector string', () => {
      const maliciousOptions = {
        tableName: 'SourceDocuments',
        vectorColumn: 'embedding',
        vectorString: "0.1'; DROP TABLE users; --",
        embeddingDim: 384,
        topK: 5
      };
      
      expect(() => {
        VectorSQLUtils.formatVectorSearchSQL(maliciousOptions);
      }).toThrow('Invalid vector string');
    });

    test('should prevent injection through additional WHERE clause', () => {
      const options = {
        tableName: 'SourceDocuments',
        vectorColumn: 'embedding',
        vectorString: '0.1,0.2,0.3',
        embeddingDim: 384,
        topK: 5,
        additionalWhere: "1=1; DROP TABLE users; --"
      };
      
      // Should throw an error to prevent SQL injection
      expect(() => {
        VectorSQLUtils.formatVectorSearchSQL(options);
      }).toThrow('Potentially dangerous SQL detected in additionalWhere clause');
    });
  });

  describe('performance and edge cases', () => {
    test('should handle very large topK values', () => {
      const options = {
        tableName: 'SourceDocuments',
        vectorColumn: 'embedding',
        vectorString: '0.1,0.2,0.3',
        embeddingDim: 384,
        topK: 10000
      };
      
      const sql = VectorSQLUtils.formatVectorSearchSQL(options);
      expect(sql).toContain('SELECT TOP 10000');
    });

    test('should handle complex vector strings', () => {
      const complexVector = Array(384).fill(0).map((_, i) => 
        Math.sin(i * 0.1).toFixed(6)
      ).join(',');
      
      expect(VectorSQLUtils.validateVectorString(complexVector)).toBe(true);
      
      const formatted = VectorSQLUtils.formatVectorForIris(
        complexVector.split(',').map(Number)
      );
      expect(formatted.split(',').length).toBe(384);
    });
  });
});