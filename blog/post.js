const progressBar = document.querySelector(".reading-progress span");
const article = document.querySelector(".post-content");

const updateReadingProgress = () => {
  if (!progressBar || !article) return;

  const articleTop = article.offsetTop;
  const articleHeight = article.offsetHeight;
  const viewportHeight = window.innerHeight;
  const distance = articleHeight - viewportHeight;
  const progress = distance > 0 ? (window.scrollY - articleTop) / distance : 1;
  const clamped = Math.min(1, Math.max(0, progress));
  progressBar.style.transform = `scaleX(${clamped})`;
};

document.querySelectorAll(".post-content table").forEach((table) => {
  const wrapper = document.createElement("div");
  wrapper.className = "table-wrap";
  table.before(wrapper);
  wrapper.append(table);
});

const tocLinks = [...document.querySelectorAll(".toc-sidebar a[href^='#']")];
const observedHeadings = tocLinks
  .map((link) => document.querySelector(link.getAttribute("href")))
  .filter(Boolean);

if ("IntersectionObserver" in window && observedHeadings.length) {
  const headingObserver = new IntersectionObserver(
    (entries) => {
      const current = entries.find((entry) => entry.isIntersecting);
      if (!current) return;

      tocLinks.forEach((link) => {
        const active = link.getAttribute("href") === `#${current.target.id}`;
        if (active) {
          link.setAttribute("aria-current", "location");
        } else {
          link.removeAttribute("aria-current");
        }
      });
    },
    { rootMargin: "-16% 0px -72%", threshold: 0 },
  );

  observedHeadings.forEach((heading) => headingObserver.observe(heading));
}

window.addEventListener("scroll", updateReadingProgress, { passive: true });
window.addEventListener("resize", updateReadingProgress);
updateReadingProgress();
