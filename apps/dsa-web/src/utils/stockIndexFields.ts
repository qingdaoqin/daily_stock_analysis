/**
 * Stock Index Field Definitions
 *
 * Define index field positions and search configuration constants
 */

/**
 * Stock index field configuration
 *
 * Define all searchable fields and their properties
 */
export const STOCK_INDEX_FIELDS = {
  canonicalCode: { label: '代码（完整）', searchable: true, weight: 10 },
  displayCode: { label: '代码', searchable: true, weight: 10 },
  nameZh: { label: '中文名称', searchable: true, weight: 8 },
  pinyinFull: { label: '全拼', searchable: true, weight: 6 },
  pinyinAbbr: { label: '拼音缩写', searchable: true, weight: 7 },
  aliases: { label: '别名', searchable: true, weight: 5 },
} as const;

/**
 * Field index in tuple format
 */
export const INDEX_FIELD = {
  CANONICAL_CODE: 0,
  DISPLAY_CODE: 1,
  NAME_ZH: 2,
  PINYIN_FULL: 3,
  PINYIN_ABBR: 4,
  ALIASES: 5,
  MARKET: 6,
  ASSET_TYPE: 7,
  ACTIVE: 8,
  POPULARITY: 9,
} as const;

/**
 * Match score weights
 */
export const MATCH_SCORE = {
  /** Exact match minimum score */
  EXACT_MIN: 96,
  /** Prefix match minimum score */
  PREFIX_MIN: 77,
  /** Contains match minimum score */
  CONTAINS_MIN: 57,
  /** Popularity bonus factor */
  POPULARITY_FACTOR: 0.1,
} as const;

/**
 * Search configuration
 */
export const SEARCH_CONFIG = {
  /** Maximum results to return */
  DEFAULT_LIMIT: 10,
  /** Minimum query length */
  MIN_QUERY_LENGTH: 1,
  /** Debounce delay ms */
  DEBOUNCE_MS: 150,
} as const;
