import { NavLink, Route, Routes } from "react-router-dom";
import Crawl     from "./pages/Crawl";
import Dataset   from "./pages/Dataset";
import Training  from "./pages/Training";
import Export    from "./pages/Export";
import Inference from "./pages/Inference";
import { useJobStore } from "./store/jobStore";
import { useTranslation } from "react-i18next";
import { useState, useEffect } from "react";

const navItems = [
  { to: "/",          icon: "⬇️",  key: "common.crawl"    },
  { to: "/dataset",   icon: "📁",  key: "common.dataset"  },
  { to: "/training",  icon: "🧠",  key: "common.train"    },
  { to: "/export",    icon: "📦",  key: "common.export"   },
  { to: "/inference", icon: "🔍",  key: "common.inference"},
];

function RunningDot({ active }: { active: boolean }) {
  if (!active) return null;
  return <span className="ml-auto w-2 h-2 rounded-full bg-blue-400 animate-pulse" />;
}

export default function App() {
  const { crawlState, trainState, quantState, onnxState } = useJobStore();
  const { t, i18n } = useTranslation();
  const [theme, setTheme] = useState(localStorage.getItem("theme") || "dark");

  useEffect(() => {
    if (theme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
    localStorage.setItem("theme", theme);
  }, [theme]);

  const toggleTheme = () => setTheme(theme === "dark" ? "light" : "dark");
  const toggleLang = () => {
    const newLang = i18n.language === "ko" ? "en" : "ko";
    i18n.changeLanguage(newLang);
  };

  return (
    <div className="flex h-screen overflow-hidden bg-white dark:bg-gray-950 text-gray-900 dark:text-gray-100 transition-colors duration-200">
      {/* ── 사이드바 ─────────────────────────────── */}
      <aside className="w-48 bg-gray-50 dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800 flex flex-col shrink-0">
        <div className="px-4 py-5 border-b border-gray-200 dark:border-gray-800">
          <span className="text-lg font-bold tracking-tight dark:text-white">{t("nav.studio")}</span>
          <p className="text-xs text-gray-500 mt-0.5">Studio</p>
        </div>

        <nav className="flex-1 py-3 space-y-1 px-2">
          {navItems.map(({ to, icon, key }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                  isActive
                    ? "bg-brand-600 text-white font-medium"
                    : "text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-800 hover:text-gray-900 dark:hover:text-white"
                }`
              }
            >
              <span>{icon}</span>
              <span>{t(key)}</span>
              {key === "common.crawl"   && <RunningDot active={crawlState === "running"} />}
              {key === "common.train"   && <RunningDot active={trainState === "running"} />}
              {key === "common.export"  && <RunningDot active={quantState === "running" || onnxState === "running"} />}
            </NavLink>
          ))}
        </nav>

        <div className="p-3 border-t border-gray-200 dark:border-gray-800 space-y-2">
          <div className="flex items-center justify-between gap-2">
            <span className="text-[10px] font-medium text-gray-500 uppercase">Settings</span>
            <div className="flex gap-1">
               <button 
                onClick={toggleLang}
                className="px-2 py-1 text-[10px] rounded bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-700 transition-colors"
              >
                {i18n.language === "ko" ? "EN" : "KO"}
              </button>
              <button 
                onClick={toggleTheme}
                className="p-1 text-xs rounded bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-700 transition-colors"
                title="Toggle Theme"
              >
                {theme === "dark" ? "☀️" : "🌙"}
              </button>
            </div>
          </div>
          <div className="text-center text-xs text-gray-400">
            v1.0.0
          </div>
        </div>
      </aside>

      {/* ── 메인 콘텐츠 ─────────────────────────── */}
      <main className="flex-1 overflow-y-auto bg-gray-50 dark:bg-gray-950">
        <Routes>
          <Route path="/"          element={<Crawl />}     />
          <Route path="/dataset"   element={<Dataset />}   />
          <Route path="/training"  element={<Training />}  />
          <Route path="/export"    element={<Export />}    />
          <Route path="/inference" element={<Inference />} />
        </Routes>
      </main>
    </div>
  );
}
