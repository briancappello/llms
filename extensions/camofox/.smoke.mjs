import { Camoufox } from "camoufox-js";

const b = await Camoufox({ headless: true, block_images: true, humanize: true });
const ctx = await b.newContext();
ctx.setDefaultTimeout(20000);
const page = await ctx.newPage();
await page.goto("https://example.com", { waitUntil: "domcontentloaded" });
console.log("TITLE:", await page.title());
console.log("URL:", page.url());
const snap = await page.ariaSnapshot({ mode: "ai" });
console.log("--- SNAPSHOT ---");
console.log(snap);
const refs = [...snap.matchAll(/\[ref=(e\d+)\]/g)].map((m) => m[1]);
console.log("REFS FOUND:", refs);
if (refs.length) {
  const t = await page.locator(`aria-ref=${refs[refs.length - 1]}`).textContent();
  console.log("RESOLVED LAST REF TEXT:", JSON.stringify(t));
}
await ctx.close();
await b.close();
console.log("OK");
