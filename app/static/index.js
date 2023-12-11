document.getElementById("go-to-profanity").addEventListener("click", function () {
  window.scrollTo({
    top: document.body.scrollHeight,
    behavior: "smooth",
  });
});

function goToHome() {
  window.location.href = "/";
}