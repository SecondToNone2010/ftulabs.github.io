// FTU Labs — Localization (l10n)
// Dropdown language switcher with auto-detection and greyed-out unavailable langs.

(function () {
  // -------------------------------------------------------------------------
  // Configuration — add new languages here
  // -------------------------------------------------------------------------

  var DEFAULT_LANG = "en";
  var STORAGE_KEY = "ftu-lang";

  // Order matters: this is the order they appear in the dropdown.
  // To add a language, append an entry here — no other JS changes needed.
  var LANGUAGES = [
    {
      code: "en",
      label: "English",
      flag:
        '<svg viewBox="0 0 20 14" xmlns="http://www.w3.org/2000/svg">' +
        '<rect width="20" height="14" rx="2" fill="#012169"/>' +
        '<path d="M0 0l20 14M20 0L0 14" stroke="#FFF" stroke-width="2.8"/>' +
        '<path d="M0 0l20 14M20 0L0 14" stroke="#C8102E" stroke-width="1.4"/>' +
        '<path d="M10 0v14M0 7h20" stroke="#FFF" stroke-width="4.6"/>' +
        '<path d="M10 0v14M0 7h20" stroke="#C8102E" stroke-width="2.8"/>' +
        "</svg>",
    },
    {
      code: "vi",
      label: "Tiếng Việt",
      flag:
        '<svg viewBox="0 0 20 14" xmlns="http://www.w3.org/2000/svg">' +
        '<rect width="20" height="14" rx="2" fill="#DA251D"/>' +
        '<polygon points="10,2.5 11.5,6.2 15.5,6.2 12.2,8.6 13.5,12.3 10,9.8 6.5,12.3 7.8,8.6 4.5,6.2 8.5,6.2" fill="#FFFF00"/>' +
        "</svg>",
    },
    // Example: to add Japanese, uncomment below and add translations + manifest entries
    // {
    //   code: "ja",
    //   label: "日本語",
    //   flag:
    //     '<svg viewBox="0 0 20 14" xmlns="http://www.w3.org/2000/svg">' +
    //     '<rect width="20" height="14" rx="2" fill="#FFF"/>' +
    //     '<circle cx="10" cy="7" r="4.2" fill="#BC002D"/>' +
    //     "</svg>",
    // },
  ];

  // Chevron SVG for the dropdown toggle
  var CHEVRON_SVG =
    '<svg class="lang-chevron" viewBox="0 0 10 6" xmlns="http://www.w3.org/2000/svg">' +
    '<path d="M1 1l4 4 4-4" stroke="currentColor" stroke-width="1.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>' +
    "</svg>";

  // Derived lookup maps
  var CODES = [];
  var BY_CODE = {};
  for (var i = 0; i < LANGUAGES.length; i++) {
    CODES.push(LANGUAGES[i].code);
    BY_CODE[LANGUAGES[i].code] = LANGUAGES[i];
  }

  // -------------------------------------------------------------------------
  // URL helpers
  // -------------------------------------------------------------------------

  function getCurrentLang() {
    var path = location.pathname;
    for (var i = 0; i < CODES.length; i++) {
      var c = CODES[i];
      if (c !== DEFAULT_LANG) {
        if (
          path === "/" + c ||
          path === "/" + c + "/" ||
          path.indexOf("/" + c + "/") === 0
        ) {
          return c;
        }
      }
    }
    return DEFAULT_LANG;
  }

  function getBasePath() {
    var path = location.pathname.replace(/^\//, "");
    var lang = getCurrentLang();
    if (lang !== DEFAULT_LANG) {
      path = path.replace(new RegExp("^" + lang + "/"), "");
    }
    return path;
  }

  function buildUrl(lang, basePath) {
    if (lang === DEFAULT_LANG) {
      return "/" + basePath;
    }
    return "/" + lang + "/" + basePath;
  }

  // -------------------------------------------------------------------------
  // Manifest URL — resolve relative to the <script> tag's own src so it
  // works from any directory depth (root pages, blog/, vi/blog/, etc.)
  // -------------------------------------------------------------------------

  var SITE_ROOT = "/";

  function manifestUrl() {
    return SITE_ROOT + "l10n/manifest.json";
  }

  // -------------------------------------------------------------------------
  // Build the custom dropdown UI inside every .lang-switch placeholder
  // -------------------------------------------------------------------------

  function buildSwitcher(manifest) {
    var currentLang = getCurrentLang();
    var basePath = getBasePath();
    var available = manifest[basePath] || [DEFAULT_LANG];
    var currentInfo = BY_CODE[currentLang] || BY_CODE[DEFAULT_LANG];

    var containers = document.querySelectorAll(".lang-switch");
    for (var c = 0; c < containers.length; c++) {
      var sw = containers[c];
      sw.innerHTML = "";

      // --- Toggle button (flag + label + chevron) ---
      var toggle = document.createElement("button");
      toggle.className = "lang-toggle";
      toggle.type = "button";
      toggle.setAttribute("aria-expanded", "false");
      toggle.setAttribute("aria-haspopup", "listbox");
      toggle.setAttribute("aria-label", "Change language");
      toggle.innerHTML =
        '<span class="lang-flag">' + currentInfo.flag + "</span>" +
        '<span class="lang-code">' + currentInfo.label + "</span>" +
        CHEVRON_SVG;

      // --- Dropdown menu ---
      var menu = document.createElement("div");
      menu.className = "lang-menu";
      menu.setAttribute("role", "listbox");

      for (var i = 0; i < LANGUAGES.length; i++) {
        var lang = LANGUAGES[i];
        var isActive = lang.code === currentLang;
        var isAvailable = available.indexOf(lang.code) !== -1;

        var option = document.createElement("a");
        option.className = "lang-option";
        option.setAttribute("role", "option");
        option.setAttribute("data-lang", lang.code);
        option.innerHTML =
          '<span class="lang-flag">' + lang.flag + "</span>" +
          '<span class="lang-label">' + lang.label + "</span>";

        if (isActive) {
          option.classList.add("active");
          option.setAttribute("aria-selected", "true");
        }

        if (isAvailable && !isActive) {
          option.href = buildUrl(lang.code, basePath);
          (function (code) {
            option.addEventListener("click", function () {
              localStorage.setItem(STORAGE_KEY, code);
            });
          })(lang.code);
        } else if (!isAvailable) {
          option.classList.add("disabled");
          option.setAttribute("aria-disabled", "true");
          option.title = lang.label + " — not available for this page";
        }

        menu.appendChild(option);
      }

      sw.appendChild(toggle);
      sw.appendChild(menu);

      // --- Toggle open/close ---
      (function (toggleBtn, container) {
        toggleBtn.addEventListener("click", function (e) {
          e.stopPropagation();
          var isOpen = container.classList.toggle("open");
          toggleBtn.setAttribute("aria-expanded", isOpen ? "true" : "false");
        });
      })(toggle, sw);
    }

    // Close all dropdowns when clicking outside
    document.addEventListener("click", function () {
      var all = document.querySelectorAll(".lang-switch.open");
      for (var i = 0; i < all.length; i++) {
        all[i].classList.remove("open");
        var btn = all[i].querySelector(".lang-toggle");
        if (btn) btn.setAttribute("aria-expanded", "false");
      }
    });

    // Close on Escape key
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") {
        var all = document.querySelectorAll(".lang-switch.open");
        for (var i = 0; i < all.length; i++) {
          all[i].classList.remove("open");
          var btn = all[i].querySelector(".lang-toggle");
          if (btn) {
            btn.setAttribute("aria-expanded", "false");
            btn.focus();
          }
        }
      }
    });
  }

  // -------------------------------------------------------------------------
  // Auto-detect language on first visit and redirect if translation exists
  // -------------------------------------------------------------------------

  function autoDetect(manifest) {
    if (localStorage.getItem(STORAGE_KEY)) {
      return; // user already made a choice
    }

    var browserLang = (navigator.language || navigator.userLanguage || "")
      .split("-")[0]
      .toLowerCase();

    var currentLang = getCurrentLang();
    localStorage.setItem(STORAGE_KEY, currentLang);

    if (browserLang !== currentLang && CODES.indexOf(browserLang) !== -1) {
      var basePath = getBasePath();
      var available = manifest[basePath] || [DEFAULT_LANG];
      if (available.indexOf(browserLang) !== -1) {
        localStorage.setItem(STORAGE_KEY, browserLang);
        location.replace(buildUrl(browserLang, basePath));
      }
    }
  }

  // -------------------------------------------------------------------------
  // Init
  // -------------------------------------------------------------------------

  function init() {
    fetch(manifestUrl())
      .then(function (r) {
        return r.json();
      })
      .then(function (manifest) {
        autoDetect(manifest);
        buildSwitcher(manifest);
      })
      .catch(function () {
        // Manifest unavailable — show switcher with only current lang active
        buildSwitcher({});
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
