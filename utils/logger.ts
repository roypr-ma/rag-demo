/**
 * Logging utilities for consistent output formatting
 */

const SEPARATOR = "=".repeat(70);
const DIVIDER = "-".repeat(70);

/**
 * Print a section header with separators
 */
export function logSection(title: string) {
  console.log("\n" + SEPARATOR);
  console.log(title);
  console.log(SEPARATOR);
}

/**
 * Print a subsection divider
 */
export function logDivider() {
  console.log(DIVIDER);
}

/**
 * Print elapsed time
 */
export function logTime(seconds: number) {
  console.log(`⏱️  Time: ${seconds.toFixed(2)}s`);
}

/**
 * Print a question being asked
 */
export function logQuestion(question: string) {
  console.log("\n" + SEPARATOR);
  console.log(`👤 Human: ${question}`);
  console.log(DIVIDER);
}

/**
 * Print a simple separator line
 */
export function logSeparator() {
  console.log(SEPARATOR);
}

/**
 * Print execution summary
 */
export function logSummary(messageCount: number) {
  console.log("\n" + SEPARATOR);
  console.log("📊 SUMMARY");
  console.log(SEPARATOR);
  console.log(`💾 Total messages in history: ${messageCount}`);
  console.log(SEPARATOR);
}

