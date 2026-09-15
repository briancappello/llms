/**
 * camofox -- headless browsing tools for the pi agent.
 *
 * WHY THIS EXISTS, AND WHY IT IS NOT jo-inc/camofox-browser
 * --------------------------------------------------------
 * pi ships read/bash/edit/write/grep/find/ls and nothing that reaches the web,
 * and it deliberately has no MCP ("It intentionally does not include built-in
 * MCP" -- pi docs/usage.md). So the only integration surface is an extension.
 *
 * The obvious candidate, jo-inc/camofox-browser, wraps Camoufox in a REST
 * server on :9377. Its headline feature is "accessibility snapshots with stable
 * element refs, ~90% smaller than raw HTML" -- but that is not its invention.
 * It is public Playwright API:
 *
 *     page.ariaSnapshot({ mode: "ai" })   -> tree with [ref=e12] markers
 *     page.locator("aria-ref=e12")        -> built-in selector engine
 *
 * Everything else that server adds is multi-tenant plumbing we do not have:
 * per-userId session isolation, MAX_SESSIONS=50, tab recycling, an HTTP hop
 * from a process that is already Node, and crash telemetry POSTed to a
 * third-party Cloudflare Worker. On a single-user box that is cost without
 * benefit, so we drive Camoufox directly and keep the two calls above.
 *
 * What we keep from that stack is the part that is actually hard: the Camoufox
 * browser binary itself, which patches Firefox at the C++ level so
 * navigator.hardwareConcurrency, WebGL renderers, AudioContext and screen
 * geometry are spoofed before any JS sees them. `camoufox-js` fetches and
 * launches it; `playwright-core` drives it.
 *
 * DESIGN NOTES
 * ------------
 * - Text-first. Of the models in config/registry.json only fable-fusion has an
 *   mmproj, and pi's default is flash-next. Screenshots would be inert for the
 *   default model, so every tool returns text and none returns an image.
 * - Aggressive truncation. An aria snapshot of a heavy page will bury a small
 *   local model, so output is capped and paginated by character offset.
 * - Lazy launch. pi extension factories run in invocations that never start a
 *   session, so nothing is spawned until the first tool call. An unref'd idle
 *   timer closes the browser after inactivity; session_shutdown closes it too.
 * - http/https only. The agent already has `read` for local files; letting a
 *   possibly prompt-injected page talk us into file:// or javascript: URLs
 *   buys nothing and costs a sandbox escape.
 *
 * Env: CAMOFOX_IDLE_MS, CAMOFOX_NAV_TIMEOUT_MS, CAMOFOX_ACTION_TIMEOUT_MS,
 *      CAMOFOX_MAX_CHARS, CAMOFOX_PROFILE_DIR, CAMOFOX_BLOCK_IMAGES,
 *      CAMOFOX_HEADLESS, CAMOFOX_HUMANIZE
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { Browser, BrowserContext, Page } from "playwright-core";
import { StringEnum } from "@earendil-works/pi-ai";
import { Type } from "typebox";

// --- configuration ---------------------------------------------------------

function envInt(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined || raw.trim() === "") return fallback;
  const n = Number(raw);
  return Number.isFinite(n) && n >= 0 ? n : fallback;
}

function envBool(name: string, fallback: boolean): boolean {
  const raw = process.env[name];
  if (raw === undefined || raw.trim() === "") return fallback;
  return !/^(0|false|no|off)$/i.test(raw.trim());
}

/** Close the browser after this long with no tool call. 0 disables. */
const IDLE_MS = envInt("CAMOFOX_IDLE_MS", 300_000);
const NAV_TIMEOUT_MS = envInt("CAMOFOX_NAV_TIMEOUT_MS", 30_000);
const ACTION_TIMEOUT_MS = envInt("CAMOFOX_ACTION_TIMEOUT_MS", 15_000);
/** Per-call character budget. ~12k chars is roughly 3k tokens. */
const MAX_CHARS = envInt("CAMOFOX_MAX_CHARS", 12_000);
/** Set to a directory to persist cookies/logins across pi sessions. */
const PROFILE_DIR = process.env.CAMOFOX_PROFILE_DIR?.trim() || undefined;
/**
 * Off by default. Blocking images is tempting here -- no model we serve can see
 * them and it would cut page weight -- but camoufox-js warns that it "has been
 * reported to cause detection issues on major WAFs", which defeats the only
 * reason to run Camoufox instead of plain Firefox. Opt in per-task if a site is
 * slow and you do not care about being flagged.
 */
const BLOCK_IMAGES = envBool("CAMOFOX_BLOCK_IMAGES", false);
const HEADLESS = envBool("CAMOFOX_HEADLESS", true);
const HUMANIZE = envBool("CAMOFOX_HUMANIZE", true);

// --- browser lifecycle -----------------------------------------------------

let browser: Browser | undefined;
let context: BrowserContext | undefined;
let page: Page | undefined;
let idleTimer: ReturnType<typeof setTimeout> | undefined;
let launching: Promise<void> | undefined;

function clearIdle(): void {
  if (idleTimer) {
    clearTimeout(idleTimer);
    idleTimer = undefined;
  }
}

function touchIdle(): void {
  clearIdle();
  if (IDLE_MS <= 0) return;
  idleTimer = setTimeout(() => {
    void closeBrowser();
  }, IDLE_MS);
  // Never hold the pi process open just because a browser is warm.
  idleTimer.unref?.();
}

async function launch(): Promise<void> {
  // camoufox-js pulls in native deps (better-sqlite3 via maxmind); importing it
  // lazily keeps pi startup cost at zero for sessions that never browse.
  const { Camoufox } = await import("camoufox-js");

  const options: Record<string, unknown> = {
    headless: HEADLESS,
    block_images: BLOCK_IMAGES,
    humanize: HUMANIZE,
  };
  // camoufox-js warns on stderr every launch when images are blocked, which
  // would smear pi's TUI. The opt-in itself is the acknowledgement.
  if (BLOCK_IMAGES) options.i_know_what_im_doing = true;

  if (PROFILE_DIR) {
    // With user_data_dir, Camoufox returns a persistent BrowserContext and
    // there is no separate Browser handle to close.
    context = (await Camoufox({
      ...options,
      user_data_dir: PROFILE_DIR,
    })) as unknown as BrowserContext;
    browser = undefined;
  } else {
    browser = (await Camoufox(options)) as unknown as Browser;
    context = await browser.newContext();
  }

  context.setDefaultTimeout(ACTION_TIMEOUT_MS);
  context.setDefaultNavigationTimeout(NAV_TIMEOUT_MS);
}

async function ensurePage(): Promise<Page> {
  // Collapse concurrent first-calls onto one launch; pi runs tools in parallel.
  if (!context) {
    launching ??= launch().finally(() => {
      launching = undefined;
    });
    await launching;
  }
  if (!page || page.isClosed()) {
    page = await context!.newPage();
  }
  touchIdle();
  return page;
}

async function closeBrowser(): Promise<void> {
  clearIdle();
  const c = context;
  const b = browser;
  context = undefined;
  browser = undefined;
  page = undefined;
  try {
    await c?.close();
  } catch {
    /* already gone */
  }
  try {
    await b?.close();
  } catch {
    /* already gone */
  }
}

/** A page must already be open for these tools to mean anything. */
function requirePage(): Page {
  if (!page || page.isClosed()) {
    throw new Error("No page is open. Call browser_navigate first.");
  }
  touchIdle();
  return page;
}

// --- guards and formatting -------------------------------------------------

/** http/https only, and tolerate the leading @ that some models emit. */
function safeUrl(raw: string): string {
  let candidate = raw.trim().replace(/^@/, "");
  if (candidate === "") throw new Error("url is empty.");
  if (!/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(candidate)) {
    candidate = `https://${candidate}`;
  }
  let parsed: URL;
  try {
    parsed = new URL(candidate);
  } catch {
    throw new Error(`Not a valid URL: ${raw}`);
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error(
      `Refusing to open a ${parsed.protocol} URL. Only http and https are allowed; ` +
        "use the read tool for local files.",
    );
  }
  return parsed.toString();
}

/** Refs come from browser_snapshot and look like e12. Never interpolate raw. */
function refLocator(target: Page, ref: string) {
  const trimmed = ref.trim();
  if (!/^e\d+$/.test(trimmed)) {
    throw new Error(
      `Invalid ref ${JSON.stringify(ref)}. Refs look like "e12" and come from browser_snapshot.`,
    );
  }
  return target.locator(`aria-ref=${trimmed}`);
}

function paginate(text: string, offset: number, label: string): string {
  const total = text.length;
  if (total === 0) return `[${label} is empty]`;
  if (offset >= total) {
    throw new Error(`offset ${offset} is past the end of the ${label} (${total} chars).`);
  }
  const slice = text.slice(offset, offset + MAX_CHARS);
  const end = offset + slice.length;
  if (end < total) {
    return `${slice}\n\n[${label} truncated: chars ${offset}-${end} of ${total}. Call again with offset=${end} for the next chunk.]`;
  }
  if (offset > 0) {
    return `${slice}\n\n[${label}: chars ${offset}-${end} of ${total}, end reached.]`;
  }
  return slice;
}

async function pageHeader(target: Page): Promise<string> {
  const title = await target.title().catch(() => "");
  return `${title || "(untitled)"}\n${target.url()}\n\n`;
}

/** Best-effort settle: DOM is enough, network idle often never arrives. */
async function settle(target: Page): Promise<void> {
  await target.waitForLoadState("domcontentloaded").catch(() => {});
}

async function ariaSnapshot(target: Page): Promise<string> {
  return await target.ariaSnapshot({ mode: "ai" });
}

async function readableText(target: Page): Promise<string> {
  const raw = await target.evaluate(() => {
    const pick =
      document.querySelector("main") ??
      document.querySelector("article") ??
      document.body;
    return pick ? (pick as HTMLElement).innerText : "";
  });
  return raw.replace(/\n{3,}/g, "\n\n").trim();
}

function text(body: string) {
  return { content: [{ type: "text" as const, text: body }], details: {} };
}

// --- extension -------------------------------------------------------------

export default function camofox(pi: ExtensionAPI) {
  pi.registerTool({
    name: "browser_navigate",
    label: "Browse",
    description:
      "Open a URL in a real headless Firefox (Camoufox) that resists bot detection, and return the page's accessibility snapshot. " +
      "Use this to read pages that plain HTTP fetching cannot reach: JavaScript-rendered apps, search results, and sites behind bot checks. " +
      "The snapshot lists interactive elements with refs like [ref=e12]; pass those refs to browser_click and browser_type. " +
      "Only http and https URLs are allowed. Output is truncated; use browser_snapshot with an offset to page through a long result.",
    promptSnippet: "Open a web page in a stealth headless browser and read it",
    promptGuidelines: [
      "Use browser_navigate when a task needs live web content; pi has no other way to reach the network.",
      "Element refs such as e12 are only valid for the page currently open, and change after browser_navigate or browser_click.",
      "Prefer browser_read over browser_snapshot when the goal is reading prose rather than interacting with controls.",
    ],
    parameters: Type.Object({
      url: Type.String({
        description: "Absolute http/https URL. A bare host like example.com is assumed to be https.",
      }),
    }),
    async execute(_id, params, signal) {
      const url = safeUrl(params.url);
      const target = await ensurePage();
      if (signal?.aborted) return text("Cancelled.");
      await target.goto(url, { waitUntil: "domcontentloaded" });
      await settle(target);
      const snapshot = await ariaSnapshot(target);
      return text((await pageHeader(target)) + paginate(snapshot, 0, "snapshot"));
    },
  });

  pi.registerTool({
    name: "browser_snapshot",
    label: "Snapshot",
    description:
      "Return the accessibility snapshot of the page that is already open, including element refs like [ref=e12] for clicking and typing. " +
      "Call this after browser_click or browser_type to see how the page changed, or with an offset to read the next chunk of a truncated snapshot. " +
      "This covers the whole page, not just the visible part.",
    promptSnippet: "Re-read the open page's structure and element refs",
    parameters: Type.Object({
      offset: Type.Optional(
        Type.Integer({
          minimum: 0,
          description: "Character offset to resume from. Omit for the start of the snapshot.",
        }),
      ),
    }),
    async execute(_id, params) {
      const target = requirePage();
      const snapshot = await ariaSnapshot(target);
      return text((await pageHeader(target)) + paginate(snapshot, params.offset ?? 0, "snapshot"));
    },
  });

  pi.registerTool({
    name: "browser_read",
    label: "Read page",
    description:
      "Extract the readable text of the page that is already open, without element refs or markup. " +
      "Use this for articles, documentation and any page whose value is its prose. " +
      "Use browser_snapshot instead when you need to click or type. Output is truncated; pass an offset to continue.",
    promptSnippet: "Extract the readable prose of the open page",
    parameters: Type.Object({
      offset: Type.Optional(
        Type.Integer({
          minimum: 0,
          description: "Character offset to resume from. Omit for the start of the text.",
        }),
      ),
    }),
    async execute(_id, params) {
      const target = requirePage();
      const body = await readableText(target);
      return text((await pageHeader(target)) + paginate(body, params.offset ?? 0, "page text"));
    },
  });

  pi.registerTool({
    name: "browser_click",
    label: "Click",
    description:
      "Click an element on the open page, identified by a ref such as e12 taken from browser_snapshot. " +
      "Returns the accessibility snapshot of the resulting page. Refs from an older snapshot may be stale; " +
      "if the click fails, call browser_snapshot again and use a fresh ref.",
    promptSnippet: "Click an element on the open page by its ref",
    parameters: Type.Object({
      ref: Type.String({ description: 'Element ref from browser_snapshot, for example "e12".' }),
    }),
    async execute(_id, params) {
      const target = requirePage();
      await refLocator(target, params.ref).click();
      await settle(target);
      const snapshot = await ariaSnapshot(target);
      return text((await pageHeader(target)) + paginate(snapshot, 0, "snapshot"));
    },
  });

  pi.registerTool({
    name: "browser_type",
    label: "Type",
    description:
      "Type text into an input, textarea or search box on the open page, identified by a ref such as e12 from browser_snapshot. " +
      "Set submit to true to press Enter afterwards, which is how you run a search. Returns the resulting accessibility snapshot.",
    promptSnippet: "Type into a field on the open page and optionally submit",
    parameters: Type.Object({
      ref: Type.String({ description: 'Element ref from browser_snapshot, for example "e12".' }),
      text: Type.String({ description: "Text to enter. Replaces any existing value." }),
      submit: Type.Optional(
        Type.Boolean({ description: "Press Enter after typing. Defaults to false." }),
      ),
    }),
    async execute(_id, params) {
      const target = requirePage();
      const locator = refLocator(target, params.ref);
      await locator.fill(params.text);
      if (params.submit) {
        await locator.press("Enter");
        await settle(target);
      }
      const snapshot = await ariaSnapshot(target);
      return text((await pageHeader(target)) + paginate(snapshot, 0, "snapshot"));
    },
  });

  pi.registerTool({
    name: "browser_scroll",
    label: "Scroll",
    description:
      "Scroll the open page to trigger lazy-loaded or infinite-scroll content, then return the new accessibility snapshot. " +
      "You do not need this to read a normal page, because browser_snapshot already covers the whole document; " +
      "use it only when content appears as you scroll.",
    promptSnippet: "Scroll the open page to load more content",
    parameters: Type.Object({
      direction: StringEnum(["down", "up"] as const, {
        description: "Direction to scroll.",
      }),
      pages: Type.Optional(
        Type.Integer({
          minimum: 1,
          maximum: 10,
          description: "How many viewport heights to scroll. Defaults to 1.",
        }),
      ),
    }),
    async execute(_id, params) {
      const target = requirePage();
      const count = params.pages ?? 1;
      const sign = params.direction === "up" ? -1 : 1;
      await target.evaluate(
        ([n, s]) => window.scrollBy(0, s * n * window.innerHeight),
        [count, sign] as const,
      );
      // Give lazy loaders a beat to append content.
      await target.waitForTimeout(750);
      const snapshot = await ariaSnapshot(target);
      return text((await pageHeader(target)) + paginate(snapshot, 0, "snapshot"));
    },
  });

  pi.registerTool({
    name: "browser_back",
    label: "Back",
    description:
      "Go back to the previous page in the open browser tab and return its accessibility snapshot. " +
      "Use this after following a link or a search result to return to the list.",
    promptSnippet: "Go back to the previous page",
    parameters: Type.Object({}),
    async execute() {
      const target = requirePage();
      const response = await target.goBack();
      if (!response) return text("No previous page in this tab's history.");
      await settle(target);
      const snapshot = await ariaSnapshot(target);
      return text((await pageHeader(target)) + paginate(snapshot, 0, "snapshot"));
    },
  });

  pi.registerTool({
    name: "browser_close",
    label: "Close browser",
    description:
      "Shut the headless browser down and free its memory. Call this when the browsing part of a task is finished. " +
      "It is not required: the browser also closes itself after a period of inactivity and when the session ends.",
    promptSnippet: "Shut down the headless browser",
    parameters: Type.Object({}),
    async execute() {
      if (!context) return text("Browser is not running.");
      await closeBrowser();
      return text("Browser closed.");
    },
  });

  pi.registerCommand("browser", {
    description: "Show camofox browser status, or close it with: /browser close",
    handler: async (args, ctx) => {
      if (args.trim() === "close") {
        await closeBrowser();
        ctx.ui.notify("camofox: browser closed", "info");
        return;
      }
      if (!context) {
        ctx.ui.notify("camofox: browser not running (starts on first tool call)", "info");
        return;
      }
      const where = page && !page.isClosed() ? page.url() : "no page open";
      const profile = PROFILE_DIR ? `persistent (${PROFILE_DIR})` : "ephemeral";
      ctx.ui.notify(`camofox: running, ${profile}, at ${where}`, "info");
    },
  });

  // Idempotent: pi may fire this more than once, and closeBrowser tolerates it.
  pi.on("session_shutdown", async () => {
    await closeBrowser();
  });
}
