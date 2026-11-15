/*
 * Save an HDB page as markdown, from inside your own browser.
 *
 * hdb.gov.sg blocks scripted clients, and its pages hide most text inside expandable sections that
 * close each other (only one can be open at a time). Run this in the browser console on a page you
 * have opened yourself: it opens each section in turn, copies its text, and downloads
 * <seed id>.md. It sends no requests of its own; it only clicks the sections you could click.
 *
 * Use:  open the page > open the browser console > paste this whole file > Enter
 * Then: move the downloaded file into manual/ and run  python crawl.py --local manual
 */
(async () => {
  // URL path -> seed id (see seeds.yaml)
  const BASE = "/managing-my-home/home-ownership/renting-out-a-flat-or-bedrooms";
  const IDS = {
    [BASE]: "hdb-owner-landing",
    [BASE + "/renting-out-a-flat/eligibility"]: "hdb-owner-flat-eligibility",
    [BASE + "/renting-out-a-flat/application-process"]: "hdb-owner-flat-application",
    [BASE + "/renting-out-a-flat/regulations"]: "hdb-owner-flat-regulations",
    [BASE + "/renting-out-a-flat/rental-statistics"]: "hdb-owner-flat-rental-statistics",
    [BASE + "/renting-out-bedrooms/eligibility"]: "hdb-owner-bedroom-eligibility",
    [BASE + "/renting-out-bedrooms/application-process"]: "hdb-owner-bedroom-application",
    [BASE + "/renting-out-bedrooms/regulations"]: "hdb-owner-bedroom-regulations",
    "/cs/infoweb/apply-manage-and-renew-rental-of-bedroom/terms-and-conditions": "hdb-owner-bedroom-terms",
    "/residential/buying-a-flat/conditions-after-buying": "hdb-conditions-after-buying",
    "/renting-a-flat/renting-from-open-market": "hdb-tenant-landing",
    "/renting-a-flat/renting-from-open-market/eligibility": "hdb-tenant-eligibility",
    "/renting-a-flat/renting-from-open-market/tenancy-matters": "hdb-tenant-tenancy-matters",
    "/renting-a-flat/renting-from-open-market/regulations": "hdb-tenant-regulations",
    "/renting-a-flat/renting-from-open-market/rental-statistics": "hdb-tenant-rental-statistics",
  };

  const SKIP =
    "script,style,noscript,svg,iframe,header,footer,nav,aside,button," +
    "[class*=Sidebar],[class*=Breadcrumb],[class*=placeholder],[class*=Cookie]";
  const BLOCK = "p,ul,ol,table,h1,h2,h3,h4,h5,h6,div,section,article,li";
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const squash = (s) => s.replace(/\s+/g, " ").trim();
  const root = document.querySelector("main") || document.body;

  // 1. Open each expandable section in turn and keep a copy of its text (opening one closes the others)
  const panels = new Map();
  let empty = 0;
  for (const btn of root.querySelectorAll("button.accordion-button")) {
    if (btn.getAttribute("aria-expanded") !== "true") btn.click();
    const panel = document.getElementById(btn.getAttribute("aria-controls"));
    for (let i = 0; i < 50 && !(panel && panel.textContent.trim()); i++) await sleep(100);
    if (panel && panel.textContent.trim()) panels.set(btn, panel.cloneNode(true));
    else empty++;
    await sleep(150);
  }

  // 2. Convert the page to markdown
  const inline = (node) => {
    let out = "";
    for (const c of node.childNodes) {
      if (c.nodeType === 3) out += c.nodeValue.replace(/\s+/g, " ");
      else if (c.nodeType === 1 && !c.matches(SKIP)) {
        const tag = c.tagName.toLowerCase();
        const inner = inline(c);
        const href = c.getAttribute("href");
        if (tag === "a" && href && inner.trim() && !href.startsWith("javascript"))
          out += `[${inner.trim()}](${new URL(href, location.href).href})`;
        else if ((tag === "strong" || tag === "b") && inner.trim()) out += `**${inner.trim()}**`;
        else if (tag === "br") out += " ";
        else out += inner;
      }
    }
    return out;
  };

  const table = (t) => {
    const rows = [...t.querySelectorAll("tr")]
      .map((tr) => [...tr.children].filter((c) => /^t[hd]$/i.test(c.tagName)).map((c) => squash(inline(c)).replace(/\|/g, "\\|")))
      .filter((r) => r.length);
    if (!rows.length) return "";
    const width = Math.max(...rows.map((r) => r.length));
    const pad = (r) => [...r, ...Array(width - r.length).fill("")];
    const line = (r) => "| " + pad(r).join(" | ") + " |";
    return [line(rows[0]), "| " + Array(width).fill("---").join(" | ") + " |", ...rows.slice(1).map(line)].join("\n");
  };

  const list = (l, depth) => {
    const out = [];
    let n = 0;
    for (const li of l.children) {
      if (li.tagName.toLowerCase() !== "li") continue;
      n++;
      const bullet = l.tagName.toLowerCase() === "ol" ? `${n}.` : "-";
      const own = document.createElement("div");
      const nested = [];
      for (const c of li.childNodes) {
        if (c.nodeType === 1 && /^(ul|ol)$/i.test(c.tagName)) nested.push(c);
        else own.appendChild(c.cloneNode(true));
      }
      out.push("    ".repeat(depth) + `${bullet} ${squash(inline(own))}`);
      for (const nl of nested) out.push(list(nl, depth + 1));
    }
    return out.join("\n");
  };

  const accordion = (item) => {
    const btn = item.querySelector("button.accordion-button");
    const out = [];
    if (btn) {
      const h = btn.querySelector("[role=heading]");
      const level = Math.min(6, Math.max(2, parseInt(h && h.getAttribute("aria-level"), 10) || 3));
      out.push("#".repeat(level) + " " + squash((h || btn).textContent));
      const desc = btn.querySelector("[class*=header-description]");
      if (desc && squash(desc.textContent)) out.push(squash(desc.textContent));
      const panel = panels.get(btn);
      if (panel) out.push(blocks(panel));
    }
    return out.filter(Boolean).join("\n\n");
  };

  // Walk children in order; consecutive inline content becomes one paragraph
  const blocks = (node) => {
    const out = [];
    let buf = document.createElement("div");
    const flush = () => {
      const t = squash(inline(buf));
      if (t) out.push(t);
      buf = document.createElement("div");
    };
    for (const c of node.childNodes) {
      if (c.nodeType === 3) {
        buf.appendChild(c.cloneNode(true));
        continue;
      }
      if (c.nodeType !== 1 || c.matches(SKIP)) continue;
      const tag = c.tagName.toLowerCase();
      if (c.matches(".accordion-item")) { flush(); out.push(accordion(c)); }
      else if (/^h[1-6]$/.test(tag) || c.getAttribute("role") === "heading") {
        flush();
        const level = tag.length === 2 && tag[0] === "h" ? +tag[1] : parseInt(c.getAttribute("aria-level"), 10) || 2;
        const t = squash(inline(c));
        if (t) out.push("#".repeat(level) + " " + t);
      } else if (tag === "table") { flush(); out.push(table(c)); }
      else if (tag === "ul" || tag === "ol") { flush(); out.push(list(c, 0)); }
      else if (c.querySelector(BLOCK)) { flush(); out.push(blocks(c)); }
      else if (tag === "p" || tag === "div" || tag === "section") { flush(); const t = squash(inline(c)); if (t) out.push(t); }
      else buf.appendChild(c.cloneNode(true));
    }
    flush();
    return out.filter(Boolean).join("\n\n");
  };

  const md = blocks(root).replace(/\n{3,}/g, "\n\n").trim() + "\n";

  // 3. Download as <seed id>.md
  const path = location.pathname.replace(/\/+$/, "");
  const id = IDS[path] || path.split("/").filter(Boolean).slice(-2).join("-");
  window.__lastMarkdown = md;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([md], { type: "text/markdown" }));
  a.download = id + ".md";
  document.body.appendChild(a);
  a.click();
  a.remove();

  const known = IDS[path] ? "" : "  (path not recognised: rename the file to its seed id)";
  console.log(`Saved ${id}.md: ${md.length} chars, ${panels.size} section(s) captured` +
    (empty ? `, ${empty} section(s) stayed EMPTY: open them by hand and run again` : "") + known);
})();
