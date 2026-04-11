/**
 * Stock Index Types
 *
 * Type definitions for stock autocomplete index
 */

/**
 * Stock index item
 */
export interface StockIndexItem {
  /** Canonical code (e.g., 000001.SZ, AAPL) */
  canonicalCode: string;
  /** Display code (e.g., 000001, AAPL) */
  displayCode: string;
  /** Chinese name */
  nameZh: string;
  /** Full pinyin */
  pinyinFull: string | null;
  /** Pinyin abbreviation */
  pinyinAbbr: string | null;
  /** Aliases */
  aliases: string[];
  /** Market (CN, HK, US, BSE, INDEX, ETF) */
  market: string;
  /** Asset type (stock, index, etf) */
  assetType: string;
  /** Whether active */
  active: boolean;
  /** Popularity score */
  popularity: number;
}

/**
 * Stock suggestion item (search result)
 */
export interface StockSuggestion extends StockIndexItem {
  /** Match score */
  score: number;
  /** Match type */
  matchType: 'exact' | 'prefix' | 'contains' | 'fuzzy';
  /** Matched field */
  matchField: string;
}

/**
 * Compressed index tuple format
 *
 * Order: [canonicalCode, displayCode, nameZh, pinyinFull, pinyinAbbr, aliases, market, assetType, active, popularity]
 */
export type StockIndexTuple = [
  string,           // canonicalCode
  string,           // displayCode
  string,           // nameZh
  string | null,    // pinyinFull
  string | null,    // pinyinAbbr
  string[],         // aliases
  string,           // market
  string,           // assetType
  boolean,          // active
  number,           // popularity
];

/**
 * Stock index data (supports both formats)
 */
export type StockIndexData = StockIndexItem[] | StockIndexTuple[];
