// rag-templates/nodejs/tests/vector_sql_utils.test.js
// Tests for VectorSQLUtils module

const { VectorSQLUtils } = require('../src/vector_sql_utils');

describe('VectorSQLUtils', () => {
  describe('validateVectorString', () => {
    test('should accept valid vector strings', () => {
      expect(VectorSQLUtils.validateVectorString('1.0,2.0,3.0')).toBe(true);
      expect(VectorSQLUtils.validateVectorString('[1.0,2.0,3.0]')).toBe(true);
      expect(VectorSQLUtils.validateVectorString('0.1, 0.2, 0.3')).toBe(true);
      expect(VectorSQLUtils.validateVectorString('-1.5,2.7,-3.9')).toBe(true);
    });

    test('should reject invalid vector strings', () => {
      expect(VectorSQLUtils.validateVectorString('invalid')).toBe(false);
      expect(VectorSQLUtils.validateVectorString('1.0;2.0;3.0')).toBe(false);
      expect(VectorSQLUtils.validateVectorString('1.0|2.0|3.0')).toBe(false);
      expect(VectorSQLUtils.validateVectorString('SELECT * FROM table')).toBe(false);
    });

    test('should handle edge cases', () => {
      expect(VectorSQLUtils.validateVectorString('')).toBe(false);
      expect(VectorSQLUtils.validateVectorString(null)).toBe(false);
      expect(VectorSQLUtils.validateVectorString(undefined)).toBe(false);
    });
  });

  describe('validateTopK', () => {
    test('should accept valid topK values', () => {
      expect(VectorSQLUtils.validateTopK(1)).toBe(true);
      expect(VectorSQLUtils.validateTopK(10)).toBe(true);
      expect(VectorSQLUtils.validateTopK(100)).toBe(true);
    });

    test('should reject invalid topK values', () => {
      expect(VectorSQLUtils.validateTopK(0)).toBe(false);
      expect(VectorSQLUtils.validateTopK(-1)).toBe(false);
      expect(VectorSQLUtils.validateTopK(1.5)).toBe(false);
      expect(VectorSQLUtils.validateTopK('10')).toBe(false);
      expect(VectorSQLUtils.validateTopK(null)).toBe(false);
      expect(VectorSQLUtils.validateTopK(undefined)).toBe(false);
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
      expect(sql).toContain('FROM SourceDocuments');
      expect(sql).toContain('VECTOR_COSINE');
      expect(sql).toContain('TO_VECTOR');
      expect(sql).toContain('ORDER BY score DESC');
      expect(sql).toContain('WHERE embedding IS NOT NULL');
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

    test('should validate table name', () => {
      const invalidOptions = {
        ...validOptions,
        tableName: 'invalid-table; DROP TABLE users;'
      };
      
      expect(() => {
        VectorSQLUtils.formatVectorSearchSQL(invalidOptions);
      }).toThrow('Invalid table name');
    });

    test('should validate vector string', () => {
      const invalidOptions = {
        ...validOptions,
        vectorString: 'invalid; DROP TABLE users;'
      };
      
      expect(() => {
        VectorSQLUtils.formatVectorSearchSQL(invalidOptions);
      }).toThrow('Invalid vector string');
    });

    test('should validate topK', () => {
      const invalidOptions = {
        ...validOptions,
        topK: -1
      };
      
      expect(() => {
        VectorSQLUtils.formatVectorSearchSQL(invalidOptions);
      }).toThrow('Invalid top_k value');
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
  });

  describe('Performance and edge cases', () => {
    test('should handle large vectors', () => {
      const largeVector = Array(1536).fill(0).map((_, i) => i / 1536);
      const formatted = VectorSQLUtils.formatVectorForIris(largeVector);
      
      expect(formatted.split(',').length).toBe(1536);
      expect(formatted).toContain('0');
      expect(formatted).toContain('0.999');
    });

    test('should handle high precision numbers', () => {
      const preciseVector = [0.123456789, -0.987654321, 0.000000001];
      const formatted = VectorSQLUtils.formatVectorForIris(preciseVector);
      
      expect(formatted).toBe('0.123456789,-0.987654321,0.000000001');
    });

    test('should generate SQL for large topK values', () => {
      const options = {
        tableName: 'SourceDocuments',
        vectorColumn: 'embedding',
        vectorString: '0.1,0.2,0.3',
        embeddingDim: 384,
        topK: 1000
      };
      
      const sql = VectorSQLUtils.formatVectorSearchSQL(options);
      expect(sql).toContain('SELECT TOP 1000');
    });
  });
});