## Parquet data layout for Next.js server (quick reference)

### Files

- Location: `data_dump/`
- Naming: `<prefix>_<timestamp>_essays.parquet`, `<prefix>_<timestamp>_prompts.parquet`, `<prefix>_<timestamp>_schools.parquet`
- Compression: snappy
- Choose the latest by timestamp.

### Schemas

- `..._essays.parquet`

  - `author_id` STRING (Essay.authorId)
  - `word_count` INT (calculated from Essay.text)
  - `created_date` TIMESTAMP (Essay.createdOn)
  - `last_modified` TIMESTAMP (Essay.updatedOn)
  - `prompt_id` STRING? (nullable)
  - `school_ids` LIST<INT> (IDs into `schools.parquet`)
  - Score columns (nullable):
    - Floats: `esslo_writing`, `esslo_detail`, `esslo_voice`, `esslo_character`, `esslo_iv`, `esslo_contribution`, `esslo_why_us`, `esslo_motivation`, `esslo_academic`, `esslo_experiences`, `esslo_reflection`
    - Ints: `score_writing`, `score_detail`, `score_voice`, `score_character`, `score_iv`, `score_contribution`, `score_why_us`, `score_motivation`, `score_academic`, `score_experiences`, `score_reflection`

- `..._prompts.parquet`

  - `prompt_id` STRING (PK)
  - `application` STRING (values like `COMMON_APP`, `COMMON_APP_ASSUMED`, `COALITION_APP`, `SUPPLEMENTAL`, `UC_APP`, `UCAS_APP`)
  - `prompt_text` STRING

- `..._schools.parquet`
  - `school_id` INT (PK)
  - `school_name` STRING

Notes on semantics:

- For `SUPPLEMENTAL` essays, `school_ids` contains only that supplemental’s school.
- For `COMMON_APP`/`COALITION_APP`, `school_ids` are all non‑UC/non‑UCAS schools the student also wrote for.
- UC/UCAS essays typically have `school_ids = []`.

### How to query (Node/Next.js with DuckDB)

- Prefer running on the Node runtime (not Edge), since DuckDB's Node bindings use native modules.
- Install: `npm i duckdb`
- Basic filter patterns:
  - Filter by application: join `essays` → `prompts` on `prompt_id`, then `WHERE application IN (...)`.
  - Filter by school: either `list_contains(school_ids, ?)` or `UNNEST` the list and join to `schools`.
  - Filter by prompt text: `WHERE lower(prompt_text) LIKE lower(?)`.
  - Filter by author: `WHERE author_id = ?`.
  - Filter by word count: `WHERE word_count BETWEEN ? AND ?`.
  - Filter by date range: `WHERE created_date >= ? AND created_date <= ?`.

Example (list_contains, no unnest):

```javascript
import Database from "duckdb";
import path from "node:path";

const db = new Database.Database(":memory:"); // or a persistent file if you prefer

function latest(prefix, dir) {
  // Implement: scan dir for files starting with prefix and pick latest timestamp
  return path.join(dir, "<prefix>_<timestamp>_essays.parquet");
}

const baseDir = "analysis_output";
const essaysPath = latest("all_essays", baseDir);
const promptsPath = latest("all_essays", baseDir).replace(
  "_essays.parquet",
  "_prompts.parquet"
);
const schoolsPath = latest("all_essays", baseDir).replace(
  "_essays.parquet",
  "_schools.parquet"
);

export async function queryEssays({
  application,
  schoolNameLike,
  promptTextLike,
  limit = 100,
}) {
  return new Promise((resolve, reject) => {
    const con = db.connect();

    const sql = `
      SELECT
        e.author_id,
        e.word_count,
        e.created_date,
        e.last_modified,
        e.prompt_id,
        p.application,
        p.prompt_text,
        e.esslo_writing, e.esslo_detail, e.esslo_voice, e.esslo_character, e.esslo_iv,
        e.esslo_contribution, e.esslo_why_us, e.esslo_motivation, e.esslo_academic,
        e.esslo_experiences, e.esslo_reflection,
        e.score_writing, e.score_detail, e.score_voice, e.score_character, e.score_iv,
        e.score_contribution, e.score_why_us, e.score_motivation, e.score_academic,
        e.score_experiences, e.score_reflection
      FROM read_parquet(?) AS e
      LEFT JOIN read_parquet(?) AS p ON p.prompt_id = e.prompt_id
      WHERE
        (? IS NULL OR p.application = ?)
        AND (? IS NULL OR EXISTS (
          SELECT 1
          FROM read_parquet(?) s
          WHERE list_contains(e.school_ids, s.school_id)
            AND lower(s.school_name) LIKE lower(?)
        ))
        AND (? IS NULL OR lower(p.prompt_text) LIKE lower(?))
      LIMIT ?;
    `;

    con.all(
      sql,
      [
        essaysPath,
        promptsPath,
        application,
        application,
        schoolNameLike,
        schoolsPath,
        `%${schoolNameLike ?? ""}%`,
        promptTextLike,
        `%${promptTextLike ?? ""}%`,
        limit,
      ],
      (err, rows) => (err ? reject(err) : resolve(rows))
    );
  });
}
```

Example (explode `school_ids` to join to names):

```sql
SELECT e.*, p.application, p.prompt_text, s.school_name
FROM read_parquet('..._essays.parquet') e
LEFT JOIN read_parquet('..._prompts.parquet') p ON p.prompt_id = e.prompt_id
LEFT JOIN read_parquet('..._schools.parquet') s
  ON s.school_id = UNNEST(e.school_ids)
WHERE p.application = 'COMMON_APP' AND s.school_name ILIKE '%stanford%';
```

### Operational guidance

- File discovery: pick the latest timestamped set per `output_filename_prefix`.
- Caching: cache result sets per `(application, school, prompt)` key; the files are immutable per timestamp.
- Performance:
  - Keep one DuckDB connection per server instance; reuse prepared statements.
  - Avoid loading entire Parquet into memory; DuckDB scans with predicate pushdown.
  - Indexing not required; Parquet + DuckDB is columnar and fast for selective queries.
- Types: score columns are nullable; handle nulls defensively in your API layer.
- Runtime: ensure routes use the Node runtime in Next.js (not Edge).
