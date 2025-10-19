---
description: Quick ticket indexing validation - check database status instantly
---

Perform instant ticket indexing validation without asking for confirmation. Execute immediately:

1. Query IRIS database (localhost:31972/USER) for current ticket count
2. Calculate progress percentage against 8,051 expected tickets
3. Check entity extraction quality (avg entities per ticket)
4. Verify embeddings are present
5. Report current indexing batch/run status from logs
6. Show ETA for completion

Provide a concise status report in this format:

```
📊 TICKET INDEXING STATUS
━━━━━━━━━━━━━━━━━━━━━━━
Progress: XXX/8,051 (X.X%)
Entities: XXX total (X.X avg/ticket)
Status: [IN PROGRESS | COMPLETE | STALLED]
ETA: ~X hours
```

No questions, no prompts - just execute and report.
