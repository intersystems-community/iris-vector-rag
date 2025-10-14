# BYOT (Bring Your Own Table) Overlay Demo

This directory contains Jupyter notebooks demonstrating how to add RAG capabilities to existing IRIS tables without data migration or duplication.

## Notebooks

### `byot_overlay_demo.ipynb`
Comprehensive demonstration of zero-copy overlay functionality on existing business tables.

**Features demonstrated:**
- Minimal configuration overlay setup
- Schema mapping capabilities  
- Zero-copy approach (no data duplication)
- Performance comparisons
- Customization options

## Requirements

```bash
pip install jupyter pandas matplotlib seaborn plotly
```

## Running the Notebooks

1. **Setup Environment**:
   ```bash
   # From project root
   cd examples/notebooks
   jupyter notebook
   ```

2. **Database Connection**:
   - Ensure IRIS database is running
   - Configure connection parameters in `.env` file
   - Run setup cells in notebook

3. **Sample Data**:
   - The notebook includes sample business data creation
   - Or use your existing IRIS table

## Sample Data

The `data/` directory contains sample business table schemas and data for demonstration purposes.

## Architecture

The overlay approach works by:

1. **Schema Detection**: Automatically detect existing table structure
2. **Minimal Configuration**: Simple mapping of text and metadata columns
3. **Zero-Copy Access**: Direct queries on existing data without duplication
4. **Vector Enhancement**: Add vector search capabilities without schema changes
5. **Backward Compatibility**: Existing applications continue to work unchanged

## Use Cases

- Adding search to document management systems
- Enhancing CRM with semantic search
- Upgrading legacy content repositories
- Modernizing knowledge bases

## Support

For questions or issues, see the main project documentation or create an issue in the repository.