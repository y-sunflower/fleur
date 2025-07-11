document.addEventListener("DOMContentLoaded", () => {
  const statisticalH1 = document.querySelector("h1#statistical-details");
  if (!statisticalH1) return;

  let sibling = statisticalH1.nextElementSibling;
  while (sibling) {
    if (["H2", "H3", "H4"].includes(sibling.tagName)) {
      sibling.classList.add("after-statistical");
    }
    sibling = sibling.nextElementSibling;
  }
});
