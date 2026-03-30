(function () {
  var CDN = "../vendor/hljs/";
  var LANGUAGES = [
    "languages/latex.min.js",
    "languages/dockerfile.min.js",
    "languages/ini.min.js",
  ];

  var COPY_SVG =
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor"><path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"/><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"/></svg>';
  var CHECK_SVG =
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';

  function addCopyButtons() {
    var pres = document.querySelectorAll(".post-content pre");
    for (var i = 0; i < pres.length; i++) {
      var pre = pres[i];
      var btn = document.createElement("button");
      btn.className = "copy-btn";
      btn.type = "button";
      btn.setAttribute("aria-label", "Copy code");
      btn.innerHTML = COPY_SVG;
      pre.style.position = "relative";
      pre.appendChild(btn);

      btn.addEventListener(
        "click",
        (function (b, p) {
          return function () {
            var code = p.querySelector("code");
            var text = code ? code.textContent : p.textContent;
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
      addCopyButtons();
    });
})();
