# Machine examples

Tasks laid out as state machines, to copy into a flows folder next to the built-in root machine.

## search

A web search the user can follow up on: "Search state machine crates for Rust", then "open the
second one", "search for the async ones instead" or "done". The task waits in `results` between
takes; an unsure follow-up stays there, and two quiet minutes end it. Opening a result asks first.

1. Register the tools in the desktop settings (`jevons-desktop.toml`). `allow` keeps them to this
   task's states:

   ```toml
   [tools.web_search]
   kind = "http"
   method = "GET"
   description = "Searches the web and returns the top results as JSON"
   url = "https://api.search.brave.com/res/v1/web/search?q={query}&count=5"
   headers = { Accept = "application/json", "X-Subscription-Token" = "${env:BRAVE_API_KEY}" }
   arguments = { query = "What to search for" }
   confirm = false                  # it only reads
   allow = ["search/*"]

   [tools.open_url]
   kind = "open"
   description = "Opens an address in the default browser"
   url = "{url}"
   arguments = { url = "The address to open" }
   allow = ["search/*"]             # confirm stays on: the bubble asks first
   ```

   Any search API that answers a GET with text works the same way, such as a SearXNG instance
   (`https://<host>/search?q={query}&format=json`).

2. Copy `search/` into the flows folder, beside `machine.fsm`.

3. Let the root machine enter it. In the flows folder's `machine.fsm`, add:

   ```text
   idle --> search : said [search]
   search --> idle
   ```

   and in its `machine.toml`, list the tools and say when a take starts a search: only when the
   words start with "Search" or "Busca", chosen with no model:

   ```toml
   tools = ["script:*", "web_search", "open_url"]

   [guards.search]
   when = { transcript = "(?i)^\\W*(search|busca)\\b" }
   prefer = { transcript = "(?i)^\\W*(search|busca)\\b" }
   ```

`jevons-desktop --check-flows` checks the folder, and the inspector's **Machines** tab shows the
task's diagram and where it is.
