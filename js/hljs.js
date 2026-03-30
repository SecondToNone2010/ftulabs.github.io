(function () {
  var CDN = "/vendor/hljs/";
  var LANGUAGES = [
    "languages/latex.min.js",
    "languages/dockerfile.min.js",
    "languages/ini.min.js",
  ];

  var COPY_SVG =
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor"><path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"/><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"/></svg>';
  var CHECK_SVG =
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';

  var SLUG_MAP = {
    sh: "gnubash",
    bash: "gnubash",
    dockerfile: "docker",
    html: "html5",
    css: "css3",
    cpp: "cplusplus",
    js: "javascript",
    yaml: "yaml",
    json: "json",
    python: "python"
  };

  var GENERIC_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>';

  function enhanceCodeBlocks() {
    var pres = document.querySelectorAll(".post-content pre");
    for (var i = 0; i < pres.length; i++) {
      var pre = pres[i];
      var code = pre.querySelector("code");
      if (!code) continue;

      pre.classList.add("has-enhanced-code");

      // Get language
      var langName = "text";
      var classList = Array.from(code.classList);
      for (var j = 0; j < classList.length; j++) {
        if (classList[j].startsWith("language-")) {
          langName = classList[j].replace("language-", "");
          break;
        }
      }

      // Create header
      var header = document.createElement("div");
      header.className = "code-header";

      var badge = document.createElement("div");
      badge.className = "lang-badge";

      var slug = SLUG_MAP[langName.toLowerCase()] || langName.toLowerCase();

      if (langName !== "text" && langName !== "plaintext") {
        var img = document.createElement("img");
        // Using explicit simpleicons request with fallback via JS onerror
        img.src = "https://cdn.simpleicons.org/" + slug + "/888888";
        img.alt = langName;
        img.onerror = function() {
          this.outerHTML = GENERIC_ICON;
        };
        badge.appendChild(img);
      } else {
        badge.innerHTML = GENERIC_ICON;
      }

      var langSpan = document.createElement("span");
      langSpan.textContent = langName;
      badge.appendChild(langSpan);

      header.appendChild(badge);

      var btn = document.createElement("button");
      btn.className = "copy-btn";
      btn.type = "button";
      btn.setAttribute("aria-label", "Copy code");
      btn.innerHTML = COPY_SVG;
      header.appendChild(btn);

      btn.addEventListener(
        "click",
        (function (b, p) {
          return function () {
            var c = p.querySelector("code");
            var text = c ? c.textContent : p.textContent;
            navigator.clipboard.writeText(text).then(function () {
              b.innerHTML = CHECK_SVG;
              b.classList.add("copied");
              setTimeout(function () {
                b.innerHTML = COPY_SVG;
                b.classList.remove("copied");
              }, 1500);
            });
          };
        })(btn, pre),
      );

      // Create body wrapper
      var bodyWrap = document.createElement("div");
      bodyWrap.className = "code-body";

      // Calculate line count
      var text = code.textContent;
      if (text.endsWith("\n")) text = text.slice(0, -1);
      var lines = text.split("\n").length;

      var lnDiv = document.createElement("div");
      lnDiv.className = "line-numbers";
      lnDiv.setAttribute("aria-hidden", "true");
      var lnHtml = "";
      for (var l = 1; l <= lines; l++) {
        lnHtml += "<span>" + l + "</span>\n";
      }
      lnDiv.innerHTML = lnHtml;

      // DOM rearrange
      pre.insertBefore(header, code);
      pre.insertBefore(bodyWrap, code);
      bodyWrap.appendChild(lnDiv);
      bodyWrap.appendChild(code);
    }
  }

  function loadScript(url) {
    return new Promise(function (resolve) {
      var s = document.createElement("script");
      s.src = url;
      s.onload = s.onerror = resolve;
      document.head.appendChild(s);
    });
  }

  // Load highlight.min.js first (languages depend on it), then all languages in parallel
  loadScript(CDN + "highlight.min.js")
    .then(function () {
      return Promise.all(
        LANGUAGES.map(function (lang) {
          return loadScript(CDN + lang);
        }),
      );
    })
    .then(function () {
      hljs.highlightAll();
      enhanceCodeBlocks();
    });
})();
