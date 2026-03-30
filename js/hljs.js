(function () {
  var CDN = "https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/";
  var queue = [
    "highlight.min.js",
    "languages/latex.min.js",
    "languages/dockerfile.min.js",
    "languages/ini.min.js",
  ];

  (function load(i) {
    if (i >= queue.length) return hljs.highlightAll();
    var s = document.createElement("script");
    s.src = CDN + queue[i];
    s.onload = s.onerror = function () {
      load(i + 1);
    };
    document.head.appendChild(s);
  })(0);
})();
