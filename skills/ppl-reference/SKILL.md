---
name: ppl-reference
description: Comprehensive PPL (Piped Processing Language) reference for OpenSearch with command syntax, functions, and examples for observability queries.
---

# PPL Language Reference

Piped Processing Language (PPL) for OpenSearch. Queries use pipe-delimited syntax starting with `source=<index>` and chaining commands with `|`. Grammar source: https://github.com/opensearch-project/sql

## Field Name Escaping

Field names containing dots, `@`, or other special characters must be enclosed in backticks:

```
`attributes.gen_ai.operation.name`, `attributes.gen_ai.usage.input_tokens`, `status.code`, `events.attributes.exception.type`, `@timestamp`
```

Critical for OTel attribute fields which use dotted naming conventions.

## API Endpoints

```
POST /_plugins/_ppl              # Execute query
POST /_plugins/_ppl/_explain     # Get execution plan

Body: {"query": "<ppl_query>"}
```

---

## Commands

### Core

**`source=<index-pattern>`** — Start a query. Alias: `search source=`.
```
source=otel-v1-apm-span-* | head 10
```

**`where <condition>`** — Filter results. Operators: `=`, `!=`, `<`, `>`, `<=`, `>=`, `AND`, `OR`, `NOT`, `LIKE`, `IN`, `BETWEEN`, `IS NULL`, `IS NOT NULL`.
```
source=otel-v1-apm-span-* | where `status.code` = 2 | head 10
```

**`fields [+|-] <field-list>`** — Select (`+`) or exclude (`-`) fields. Default is `+`.
```
source=otel-v1-apm-span-* | fields traceId, spanId, serviceName, durationInNanos | head 10
```

**`stats <aggregation>... [by <field-list>]`** — Aggregate. Functions: `count()`, `sum()`, `avg()`, `max()`, `min()`, `var_samp()`, `var_pop()`, `stddev_samp()`, `stddev_pop()`, `distinct_count()`, `percentile(field, pct)`, `earliest()`, `latest()`, `list()`, `values()`, `first()`, `last()`.
```
source=otel-v1-apm-span-* | stats count() as span_count, avg(durationInNanos) as avg_duration by serviceName
```

**`sort [+|-] <field>`** — `+` ascending (default), `-` descending.
```
source=otel-v1-apm-span-* | sort - durationInNanos | head 10
```

**`head [N]`** — Limit (default N=10).
```
source=otel-v1-apm-span-* | head 5
```

**`eval <field> = <expression>`** — Compute new fields.
```
source=otel-v1-apm-span-* | eval duration_ms = durationInNanos / 1000000 | fields traceId, serviceName, duration_ms | sort - duration_ms | head 10
```

**`dedup [N] <field-list> [keepempty=<bool>] [consecutive=<bool>]`** — Remove duplicates.

> **Caveat:** `dedup` may throw a ClassCastException on fields with mixed types. Ensure consistent type.

```
source=otel-v1-apm-span-* | dedup serviceName | fields serviceName
```

**`rename <old> AS <new>`** — Rename fields.
```
source=otel-v1-apm-span-* | rename serviceName as service, durationInNanos as duration | fields traceId, service, duration | head 10
```

**`top [N] <field> [by <group>]`** — Most frequent values.
```
source=otel-v1-apm-span-* | top 5 serviceName
```

**`rare <field> [by <group>]`** — Least frequent values.
```
source=otel-v1-apm-span-* | rare `attributes.gen_ai.operation.name`
```

**`table <field-list>`** — Tabular display (alias for `fields` in some contexts).
```
source=otel-v1-apm-span-* | where `status.code` = 2 | table traceId, spanId, serviceName, name | head 10
```

**`reverse`** — Reverse the order of results.
```
source=otel-v1-apm-span-* | sort startTime | head 20 | reverse
```

### Time-Series

**`timechart span=<interval> <aggregation>... [by <field>]`** — Time-bucketed aggregation. Rate functions: `per_second()`, `per_minute()`, `per_hour()`, `per_day()`.
```
source=otel-v1-apm-span-* | timechart span=5m count() as span_count by serviceName
source=otel-v1-apm-span-* | timechart span=1h per_minute(count()) as spans_per_min by serviceName
```

**`chart <aggregation>... by <field>`** — General charting.
```
source=otel-v1-apm-span-* | chart avg(durationInNanos) by serviceName
```

**`bin`** / **`span(<field>, <interval>)`** — Bucket values. See Span Expression section below.
```
source=otel-v1-apm-span-* | stats count() by span(durationInNanos, 1000000000)
```

**`trendline [sort <field>] sma(<period>, <field>) [as <alias>]`** — Simple moving average.
```
source=otel-v1-apm-span-* | trendline sort startTime sma(10, durationInNanos) as avg_duration | fields startTime, durationInNanos, avg_duration | head 50
```

**`streamstats <aggregation>... [by <field>]`** — Running cumulative stats.

> **Caveat:** Processes all rows in memory. Always add `| head N` before `streamstats` to limit data volume.

```
source=otel-v1-apm-span-* | sort startTime | head 50 | streamstats count() as running_count, sum(`attributes.gen_ai.usage.input_tokens`) as cumulative_tokens | fields startTime, running_count, cumulative_tokens
```

**`eventstats <aggregation>... [by <field>]`** — Add aggregation as new field without collapsing rows.

> **Caveat:** Processes all rows in memory. Always add `| head N` before `eventstats`.

```
source=otel-v1-apm-span-* | head 100 | eventstats avg(durationInNanos) as avg_svc_duration by serviceName | eval deviation = durationInNanos - avg_svc_duration | fields traceId, serviceName, durationInNanos, avg_svc_duration, deviation | sort - deviation | head 20
```

### Parse / Extract

**`parse <field> '<regex-with-named-groups>'`** — Extract fields via regex.

> **Caveat:** May silently drop extracted fields on some OpenSearch versions. Use `grok` or `rex` if `parse` misbehaves.

```
source=logs-otel-v1-* | parse body '(?P<level>\w+): (?P<msg>.+)' | fields level, msg | head 10
```

**`grok <field> '<grok-pattern>'`** — Extract via named Grok patterns.

> **Caveat:** Processes all rows in memory. Always add `| head N` before `grok`.

```
source=logs-otel-v1-* | head 100 | grok body '%{LOGLEVEL:level} %{GREEDYDATA:message}' | fields level, message | head 10
```

**`rex field=<field> '<regex>'`** — Splunk-compatible regex extract.
```
source=logs-otel-v1-* | rex field=body '(?<statuscode>\d{3})' | fields statuscode, body | head 10
```

**`regex`** — Filter results using a regular expression match on a field (used within `where`).
```
source=logs-otel-v1-* | where body like '%error%' | fields traceId, body, severityText | head 10
```

**`patterns <field>`** — Auto-cluster similar log messages.
```
source=logs-otel-v1-* | patterns body | fields body, patterns_field | head 20
```

**`spath input=<field> [path=<path>] [output=<field>]`** — Extract from structured data (JSON/XML).

> **Note:** Verify the target field exists (`describe <index>`) before using `spath`.

```
source=otel-v1-apm-span-* | where isnotnull(`attributes.gen_ai.tool.name`) | spath input=`attributes.gen_ai.tool.name` | head 10
```

### Join / Lookup / Subquery

**`join left=<alias> right=<alias> ON <condition> <right-source>`** — Types: `inner`, `left`, `right`, `cross`.
```
source=otel-v1-apm-span-* | join left=s right=l ON s.traceId = l.traceId logs-otel-v1-* | fields s.spanId, s.name, l.severityText, l.body | head 10
```

**`lookup <lookup-index> <match-field> [AS <alias>] [OUTPUT <field-list>]`** — Enrich from another index.

> **Note:** The service map index (`otel-v2-apm-service-map`) uses `sourceNode`/`targetNode`, not `serviceName`.

```
source=otel-v1-apm-span-* | lookup otel-v2-apm-service-map serviceName AS `sourceNode` | fields serviceName, `targetNode`, durationInNanos | head 10
```

**`graphlookup <index> connectFromField=<f> connectToField=<f> [maxDepth=<N>]`** — Graph traversal.

> **Caveat:** Limited support in OpenSearch 3.x PPL. Test before relying on this.

```
source=otel-v2-apm-service-map | graphlookup otel-v2-apm-service-map connectFromField=`destination.domain` connectToField=serviceName maxDepth=3 as dependencies | head 10
```

**`where <field> IN [ source=<index> | ... | fields <field> ]`** — Subquery filter.
```
source=otel-v1-apm-span-* | where traceId IN [ source=otel-v1-apm-span-* | where `status.code` = 2 | fields traceId ] | fields traceId, spanId, serviceName, name | head 20
```

**`append [ source=<index> | ... ]`** — Append rows from another query.
```
source=otel-v1-apm-span-* | stats count() as cnt by serviceName | append [ source=logs-otel-v1-* | stats count() as cnt by `resource.attributes.service.name` ] | head 20
```

**`appendcol [ <commands> ]`** — Append columns from a sub-pipeline.

> **Caveat:** `source=` is NOT valid inside `appendcol[]` — it operates on the current result set. Use `append` if you need data from another index.

```
source=otel-v1-apm-span-* | stats count() as span_count, avg(durationInNanos) as avg_dur | appendcol [ stats max(durationInNanos) as max_dur ]
```

**`appendpipe [ <commands> ]`** — Append results of sub-pipeline on current data.
```
source=otel-v1-apm-span-* | stats count() as cnt by serviceName | appendpipe [ stats sum(cnt) as total ]
```

### Transform

**`fillnull [with <value>] [<field-list>]`** — Replace nulls.

> **Caveat:** Backtick-quoted field names are NOT supported in `fillnull` field list. Use `eval` to rename dotted fields first, or apply without a field list.

```
source=otel-v1-apm-span-* | eval tokens = `attributes.gen_ai.usage.input_tokens` | fillnull with 0 tokens
```

**`flatten <field>`** — Flatten nested fields to top level.
```
source=otel-v1-apm-span-* | flatten events | head 10
```

**`expand <field>`** / **`mvexpand <field>`** — Expand array/multi-value fields into separate rows.
```
source=otel-v1-apm-span-* | expand events | fields traceId, spanId, events | head 20
source=otel-v1-apm-span-* | mvexpand events | fields traceId, spanId, events | head 20
```

**`transpose [<N>]`** — Pivot rows into columns.
```
source=otel-v1-apm-span-* | stats count() as cnt by serviceName | transpose
```

**`mvcombine <field>`** — Combine rows with same key into multi-value field.
```
source=otel-v1-apm-span-* | fields traceId, serviceName | mvcombine serviceName | head 10
```

**`nomv <field>`** — Multi-value → single-value.

> **Caveat:** Only works on string arrays. Use `flatten` or `expand` for nested object arrays.

```
source=otel-v1-apm-span-* | nomv events | fields traceId, events | head 10
```

**`convert <function>(<field>) [as <alias>]`** — Type conversion. Functions: `auto()`, `num()`, `ip()`, `ctime()`, `dur2sec()`, `mktime()`, `mstime()`, `rmcomma()`, `rmunit()`, `memk()`, `none()`.
```
source=otel-v1-apm-span-* | eval duration_str = CAST(durationInNanos AS STRING) | convert num(duration_str) as duration_num | fields traceId, duration_num | head 10
```

**`eval <field> = replace(<field>, '<old>', '<new>')`** — Replace values via `replace()` string function.
```
source=logs-otel-v1-* | eval severityText = replace(severityText, 'ERROR', 'ERR') | fields severityText, body | head 10
```

### Totals

**`addcoltotals [<field-list>]`** — Add summary row with column totals.
```
source=otel-v1-apm-span-* | stats count() as cnt by serviceName | addcoltotals
```

**`addtotals [row=<bool>] [col=<bool>] [<field-list>]`** — Add row with sum of numeric fields.
```
source=otel-v1-apm-span-* | stats sum(`attributes.gen_ai.usage.input_tokens`) as input_tok, sum(`attributes.gen_ai.usage.output_tokens`) as output_tok by serviceName | addtotals
```

### ML

**`ad [time_field=<field>] [number_of_trees=<N>] [shingle_size=<N>] [time_zone=<tz>]`** — Anomaly detection.

> **Note:** `ad` takes no positional field argument; it auto-detects from preceding `stats`/`eval` output.

```
source=otel-v1-apm-span-* | where durationInNanos > 0 | ad time_field=startTime number_of_trees=100 time_zone="UTC" | head 50
```

**`kmeans [centroids=<N>] [iterations=<N>] [distance_type=<type>]`** — K-means clustering.

> **Note:** No positional field args; operates on all numeric fields from preceding output. Use `fields` to control input.

```
source=otel-v1-apm-span-* | where durationInNanos > 0 | fields traceId, serviceName, durationInNanos | kmeans centroids=3 | fields traceId, serviceName, durationInNanos, ClusterID | head 30
```

**`ml action=<algorithm>`** — General ML command. Supported: `kmeans`, `ad`.

> **Note:** `ml action=rcf` is NOT valid in OpenSearch 3.x. Use `ad` directly for Random Cut Forest.

```
source=otel-v1-apm-span-* | where durationInNanos > 0 | ml action=kmeans centroids=3 | head 50
```

### System Commands

> **Caveat:** `describe` and `show datasources` are **standalone** top-level commands, NOT pipe commands. They cannot appear after `|`.
> - ✓ `describe my-index-*`
> - ✗ `source=my-index-* | describe`

```
describe otel-v1-apm-span-*       # Inspect index mapping/field types
show datasources                   # List PPL data sources
```

### Display

**`fieldformat <field> = <format-expr>`** — Format display without changing data.
```
source=otel-v1-apm-span-* | eval duration_ms = durationInNanos / 1000000 | fieldformat duration_ms = CONCAT(CAST(duration_ms AS STRING), ' ms') | fields traceId, serviceName, duration_ms | head 10
```

---

## Span Expression

Buckets numeric or datetime values into intervals. Used with `stats`, `timechart`, `chart`.

**Syntax**: `span(<field>, <interval>)`

**Time units**: `ms`, `s`, `m` (minutes), `h`, `d`, `w`, `M` (months), `q` (quarters), `y`.

```
source=otel-v1-apm-span-* | stats count() as span_count, avg(durationInNanos) as avg_duration by span(startTime, 1h)
source=otel-v1-apm-span-* | stats count() by span(durationInNanos, 1000000000)   # numeric, plain number
```

---

## Functions

### Aggregation (for stats, eventstats, streamstats, timechart, chart)

| Function | Description |
|----------|-------------|
| `count()`, `sum(f)`, `avg(f)`, `max(f)`, `min(f)` | Basic aggregations |
| `var_samp(f)`, `var_pop(f)`, `stddev_samp(f)`, `stddev_pop(f)` | Variance/stddev |
| `distinct_count(f)` | Count distinct values |
| `percentile(f, pct)` | Value at percentile |
| `earliest(f)`, `latest(f)` | Chronological first/last |
| `first(f)`, `last(f)` | Result-order first/last |
| `list(f)`, `values(f)` | All / distinct values as list |
| `covar_pop(f1, f2)`, `covar_samp(f1, f2)` | Covariance |

```
source=otel-v1-apm-span-* | stats count() as total, avg(durationInNanos) as avg_ns, percentile(durationInNanos, 95) as p95_ns, distinct_count(serviceName) as services
```

> **Note:** `corr()` is NOT supported in OpenSearch 3.x PPL. Use `covar_samp` with separate `stddev` calls to approximate Pearson correlation.

```
source=otel-v1-apm-span-* | where `attributes.gen_ai.usage.input_tokens` > 0 | stats covar_samp(`attributes.gen_ai.usage.input_tokens`, durationInNanos) as token_duration_covar
```

### Condition

| Function | Description |
|----------|-------------|
| `isnull(f)`, `isnotnull(f)` | Null check |
| `if(cond, true_val, false_val)` | Conditional |
| `ifnull(f, default)`, `nullif(a, b)`, `coalesce(v1, v2, ...)` | Null handling |
| `case(cond1, val1, cond2, val2, ..., else_val)` | Multi-branch |
| `field LIKE 'pattern'`, `field IN (v1, ...)`, `field BETWEEN a AND b` | Used in `where` |

```
source=otel-v1-apm-span-* | eval status_label = case(`status.code` = 0, 'UNSET', `status.code` = 1, 'OK', `status.code` = 2, 'ERROR') | stats count() by status_label
```

### Conversion

| Function | Description |
|----------|-------------|
| `cast(f AS type)` | Cast (STRING, INT, LONG, FLOAT, DOUBLE, BOOLEAN, DATE, TIMESTAMP) |
| `tostring(f)`, `tonumber(f)`, `toint(f)`, `tolong(f)`, `tofloat(f)`, `todouble(f)`, `toboolean(f)` | Type shortcuts |

```
source=otel-v1-apm-span-* | eval duration_ms = CAST(durationInNanos AS DOUBLE) / 1000000.0 | fields traceId, serviceName, duration_ms | sort - duration_ms | head 10
```

### Datetime

| Function | Description |
|----------|-------------|
| `now()`, `curdate()`, `curtime()`, `sysdate()` | Current time |
| `utc_date()`, `utc_time()`, `utc_timestamp()` | UTC variants |
| `date_format(date, fmt)` | Format (`%Y-%m-%d %H:%i:%s`) |
| `date_add(d, INTERVAL n unit)`, `date_sub(d, INTERVAL n unit)`, `adddate`, `subdate` | Add/subtract |
| `datediff(d1, d2)` | Days between |
| `timestampadd(unit, n, ts)`, `timestampdiff(unit, t1, t2)` | Precise delta |
| `day()`, `month()`, `year()`, `hour()`, `minute()`, `second()` | Extract components |
| `dayofweek()`, `dayofyear()`, `week()` | Calendar components |
| `unix_timestamp(d)`, `from_unixtime(epoch)` | Epoch conversion |
| `maketime(h, m, s)`, `makedate(year, doy)` | Construct |
| `period_add(p, n)`, `period_diff(p1, p2)` | YYMM/YYYYMM arithmetic |

```
source=otel-v1-apm-span-* | where startTime > DATE_SUB(NOW(), INTERVAL 1 HOUR) | stats count() as recent_spans by serviceName
```

### String

| Function | Description |
|----------|-------------|
| `concat(...)`, `length(s)`, `char_length(s)`, `octet_length(s)`, `bit_length(s)` | Basic |
| `lower(s)`, `upper(s)`, `reverse(s)` | Case / reverse |
| `trim(s)`, `ltrim(s)`, `rtrim(s)` | Trim |
| `substring(s, start [, len])`, `substr`, `mid`, `left(s, n)`, `right(s, n)` | Substring |
| `replace(s, from, to)`, `regexp_replace(s, pattern, repl)` | Replace |
| `locate(sub, s [, pos])`, `position(sub IN s)`, `instr(s, sub)` | Search |
| `lpad(s, len, pad)`, `rpad(s, len, pad)`, `space(n)`, `repeat(s, n)` | Padding |
| `strcmp(a, b)`, `ascii(s)`, `format(val, decimals)` | Misc |
| `regexp(s, pat)`, `regexp_extract(s, pat [, group])` | Regex (returns 0/1 or group) |
| `field(s, v1, ...)`, `find_in_set(s, list)` | Position in list |
| `insert(s, pos, len, new)` | Insert |

```
source=logs-otel-v1-* | eval body_lower = lower(body) | where body_lower like '%exception%' | eval short_body = left(body, 200) | fields traceId, severityText, short_body | head 10
```

### Math

| Function | Description |
|----------|-------------|
| `abs`, `ceil`, `floor`, `round(v [, dec])`, `truncate(v, dec)`, `sign` | Rounding / sign |
| `sqrt`, `pow(b, e)`, `exp`, `ln`, `log`, `log2`, `log10` | Power / log |
| `mod(a, b)`, `conv(v, from, to)`, `crc32` | Misc |
| `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2(y, x)`, `cot` | Trig |
| `degrees(r)`, `radians(d)`, `pi()`, `e()`, `rand([seed])` | Constants / conversion |

```
source=otel-v1-apm-span-* | eval duration_ms = round(durationInNanos / 1000000.0, 2) | where duration_ms > 0 | fields traceId, serviceName, duration_ms | sort - duration_ms | head 10
```

### Collection / Multi-Value

| Function | Description |
|----------|-------------|
| `array(v1, v2, ...)`, `split(f, delim)` | Construct |
| `mvcount(f)`, `mvindex(f, i)`, `mvfirst(f)`, `mvlast(f)` | Access |
| `mvappend(f1, f2)`, `mvjoin(f, delim)`, `mvzip(f1, f2, delim)` | Combine |
| `mvdedup(f)`, `mvsort(f)`, `mvfilter(expr)` | Transform |
| `mvrange(start, end, step)` | Generate |

```
source=otel-v1-apm-span-* | eval tokens = array(`attributes.gen_ai.usage.input_tokens`, `attributes.gen_ai.usage.output_tokens`) | fields traceId, tokens | head 10
```

### JSON

| Function | Description |
|----------|-------------|
| `json_extract(f, path)`, `json_extract_path_text(f, path)` | Extract |
| `json_keys(f)`, `json_valid(f)`, `json_array_length(f)` | Inspect |
| `json_array(v1, ...)`, `json_object(k1, v1, ...)`, `to_json_string(f)` | Construct |

```
source=otel-v1-apm-span-* | where json_valid(`attributes.gen_ai.tool.call.arguments`) | eval tool_args = json_extract(`attributes.gen_ai.tool.call.arguments`, '$') | fields traceId, `attributes.gen_ai.tool.name`, tool_args | head 10
```

### Crypto / IP / System

| Function | Description |
|----------|-------------|
| `md5(f)`, `sha1(f)`, `sha2(f, numBits)` | Hash (numBits: 224/256/384/512) |
| `cidrmatch(ip, 'cidr')` | CIDR range check |
| `geoip(ip)` | Geo lookup (country, region, city, lat/lon) |
| `typeof(f)` | Data type of value |

```
source=otel-v1-apm-span-* | eval trace_hash = md5(traceId) | fields traceId, trace_hash | head 5
source=otel-v1-apm-span-* | eval type_of_duration = typeof(durationInNanos) | fields traceId, durationInNanos, type_of_duration | head 5
```

### Relevance (Full-Text Search)

| Function | Description |
|----------|-------------|
| `match(field, query)`, `match_query(...)` | Full-text match |
| `match_phrase(field, phrase)`, `match_phrase_prefix(...)`, `match_bool_prefix(...)` | Phrase / prefix |
| `multi_match([f1, f2], q)` | Multi-field match |
| `query_string([f1, f2], q)`, `simple_query_string(...)` | Lucene/simplified query syntax |
| `wildcard_query(f, pattern)` | `*` and `?` wildcards |
| `highlight(f)`, `score(rf)`, `scorequery(rf)` | Highlighting / scoring |

```
source=logs-otel-v1-* | where match(body, 'timeout error') | fields traceId, severityText, body | head 10
```

### Expressions / Operators

- **Arithmetic**: `+`, `-`, `*`, `/`
- **Comparison**: `=`, `!=` or `<>`, `<`, `>`, `<=`, `>=`
- **Logical**: `AND`, `OR`, `NOT`, `XOR`

```
source=otel-v1-apm-span-* | eval duration_ms = durationInNanos / 1000000, total_tokens = `attributes.gen_ai.usage.input_tokens` + `attributes.gen_ai.usage.output_tokens` | where duration_ms > 1000 AND total_tokens > 0 | fields traceId, serviceName, duration_ms, total_tokens | head 10
```

---

## References

- [Official PPL docs](https://github.com/opensearch-project/sql/blob/main/docs/user/ppl/index.md) — Fetch if queries fail due to OpenSearch version differences or new syntax.
