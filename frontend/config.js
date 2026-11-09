// Backend URL. Local dev -> uvicorn on 8000. Production -> set to YOUR deployed backend (no trailing slash).
// NOTE: the default below is the URL found in a comment of the original script.js; verify it is your live backend.
window.MHP_CONFIG = {
  API_BASE: (["localhost", "127.0.0.1", ""].includes(location.hostname))
    ? "http://127.0.0.1:8000"
    : "https://mansik-santulan-score.onrender.com"
};
