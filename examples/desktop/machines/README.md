# Machine examples

Agents and their tasks, laid out as state machines, to copy into a flows folder next to the
built-in agents.

## research

An agent that searches the web, with one task, `search`, that the user can follow up on: "Search
state machine crates for Rust", then "open the second one", "search for the async ones instead"
or "done". Each search is a task of its own: it waits in `results` between takes, an unsure
follow-up leaves it there, and two quiet minutes end it. Opening a result asks first. The agent
starts a search only on "Search …" or "Busca …"; anything else is for it only while a search
waits.

Both machines set `unsure = "parent"`: words the search does not know what to do with go back to
the agent, and from there to the root, so what you dictate while a search waits is still typed.
The search's diagram in the Machines tab then offers its transitions, should the words have been
for it after all.

1. Register the tools, each in the settings file of the side that runs it. `allow` keeps them to
   this task's states. The search runs with the server, in `jevons-server.toml`:

   ```toml
   [tools.web_search]
   kind = "http"
   method = "GET"
   description = "Searches the web and returns the top results as JSON"
   url = "https://api.search.brave.com/res/v1/web/search?q={query}&count=5"
   headers = { Accept = "application/json", "X-Subscription-Token" = "${env:BRAVE_API_KEY}" }
   arguments = { query = "What to search for" }
   confirm = false                  # it only reads
   allow = ["research/search/*"]
   ```

   Opening a result happens on your machine, in `jevons-desktop.toml`:

   ```toml
   [tools.open_url]
   kind = "open"
   description = "Opens an address in the default browser"
   url = "{url}"
   arguments = { url = "The address to open" }
   allow = ["research/search/*"]             # confirm stays on: the bubble asks first
   ```

   Any search API that answers a GET with text works the same way, such as a SearXNG instance
   (`https://<host>/search?q={query}&format=json`). To try it with no key, search English
   Wikipedia's articles:
   `https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch={query}&gsrlimit=5&prop=info&inprop=url&format=json&formatversion=2`,
   with a `User-Agent` header, since Wikipedia refuses requests without one.

2. Copy `research/` into the flows folder, beside `root.fsm`.

3. Let the root hand it takes. In the flows folder's `root.fsm`, add:

   ```text
   idle --> research : said
   research --> idle
   ```

   and in `root.toml`, add its tools to the list, which covers everything the agents may call:

   ```toml
   tools = ["script:*", "web_search", "open_url"]
   ```

   The agent's own files say the rest: `research/agent.toml` holds its description, its tools
   and the rule that starts a search, and `research/agent.fsm` its one state, `search`, whose
   folder is the task.

`jevons-desktop --check-flows` checks the folder, and the inspector's **Machines** tab shows the
agent, its running searches and each one's diagram.
