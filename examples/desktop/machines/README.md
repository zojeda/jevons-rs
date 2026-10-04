# Machine examples

Agents and their tasks, laid out as state machines, to copy into a flows folder next to the
built-in agents.

## research

An agent that looks things up on the internet, with one task, `search`, that the user can follow
up on: "Busca crates de máquinas de estado para Rust" or "Noticias de Paraná", then "leeme la
segunda", "open the first one", "search for the async ones instead" or "done". Each search is a
task of its own: it waits in `results` between takes, an unsure follow-up leaves it there, and
two quiet minutes end it.

- **Where it searches.** Words that name the news ("noticias", "últimas", "hoy", "this week")
  search the news with no model; otherwise the decision model chooses between the news and the
  web by their descriptions (`search/searching/`).
- **Reading a result.** "Leeme …", "resumime …" or "read …" fetches that page and says what it
  says in the bubble (`search/reading/`). Opening a result in the browser asks first.
- **Starting one.** Only on a search word ("Search …", "Busca …", "Buscame …") or a request for
  news ("Noticias de …", "Dame las últimas noticias …"), after at most two words of lead-in.
  Anything else is for the agent only while a search waits. While one waits, the same words
  search again in it: one search at a time.

Both machines set `unsure = "parent"`: words the search does not know what to do with go back to
the agent, and from there to the root, so what you dictate while a search waits is still typed.
The search's diagram in the Machines tab then offers its transitions, should the words have been
for it after all.

1. Register the tools, each in the settings file of the side that runs it. `allow` keeps them to
   this task's states. The searches and the reader run with the server, in `jevons-server.toml`.
   These use [Tavily](https://tavily.com) (one endpoint with a web and a news mode, short
   results made for language models; its free plan is 1,000 searches a month) and
   [Jina Reader](https://jina.ai/reader) (a page as clean text, no key). What you search for goes
   to Tavily, and the address of each page you have read goes to Jina, which fetches it.

   ```toml
   [tools.web_search]
   kind = "http"
   description = "Searches the web and returns the top results as JSON, each with its title, address (url) and a summary (content)"
   url = "https://api.tavily.com/search"
   headers = { Authorization = "Bearer ${env:TAVILY_API_KEY}" }
   # `country` puts that country's pages first; it is for the web search only.
   body = '{"query": "{query}", "topic": "general", "max_results": 5, "country": "argentina"}'
   arguments = { query = "What to search for" }
   confirm = false                  # it only reads
   allow = ["research/search/*"]

   [tools.news_search]
   kind = "http"
   description = "Searches the news of the last week and returns the top results as JSON, each with its title, address (url), date (published_date) and a summary (content)"
   url = "https://api.tavily.com/search"
   headers = { Authorization = "Bearer ${env:TAVILY_API_KEY}" }
   body = '{"query": "{query}", "topic": "news", "max_results": 5, "time_range": "week"}'
   arguments = { query = "What to search the news for" }
   confirm = false
   allow = ["research/search/*"]

   [tools.read_page]
   kind = "http"
   method = "GET"
   description = "Reads a web page and returns its text"
   url = "https://r.jina.ai/{url}"
   headers = { "X-Retain-Images" = "none", "X-Md-Link-Style" = "discarded" }
   arguments = { url = "The address of the page to read" }
   timeout_s = 30
   max_output = 8000                # what the model can read with room to answer
   confirm = false
   allow = ["research/search/*"]
   ```

   The key is read from the environment when a search runs, so it stays out of the file: set
   `TAVILY_API_KEY` for your user (on Windows, `setx TAVILY_API_KEY "tvly-…"` in a terminal) and
   start jevons again. Without it a search fails, and the trace says the key is missing.

   Opening a result happens on your machine, in `jevons-desktop.toml`:

   ```toml
   [tools.open_url]
   kind = "open"
   description = "Opens an address in the default browser"
   url = "{url}"
   arguments = { url = "The address to open" }
   allow = ["research/search/*"]             # confirm stays on: the bubble asks first
   ```

   Any search API that answers with JSON or text works the same way: Brave Search
   (`https://api.search.brave.com/res/v1/web/search?q={query}&count=5` and `/news/search`, with
   `X-Subscription-Token`), a SearXNG instance (`https://<host>/search?q={query}&format=json`),
   or, with no key and for facts only, English Wikipedia's articles:
   `https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch={query}&gsrlimit=5&prop=info%7Cextracts&inprop=url&exintro=1&explaintext=1&exsentences=3&exlimit=5&format=json&formatversion=2`,
   with a `User-Agent` header, since Wikipedia refuses requests without one. An answer needs
   something to answer from: results with a summary each, not titles and addresses alone.

2. Copy `research/` into the flows folder, beside `root.fsm`.

3. Let the root hand it takes. In the flows folder's `root.fsm`, add:

   ```text
   idle --> research : said
   research --> idle
   ```

   and in `root.toml`, add its tools to the list, which covers everything the agents may call:

   ```toml
   tools = ["script:*", "web_search", "news_search", "read_page", "open_url"]
   ```

   The root's `question` there names two takers, the application and the assistant. Add the
   third, so that a follow-up with none of the search words ("open the second one") is read as
   one:

   ```toml
   question = """
   What does the user want done with what they said? Decide who the words are for: the application \
   (text to type there, including questions meant for the people they write to), you, the \
   assistant listening (a question to answer for them), or something you are doing for them that \
   waits for what they say next, such as a search whose results are shown."""
   ```

   The agent's own files say the rest: `research/agent.toml` holds its description, its tools
   and the rule that starts a search, and `research/agent.fsm` its one state, `search`, whose
   folder is the task.

`jevons-desktop --check-flows` checks the folder, and the inspector's **Machines** tab shows the
agent, its running searches and each one's diagram.
