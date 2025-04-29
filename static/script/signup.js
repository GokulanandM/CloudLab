// Validate password and re-entered password match
document.querySelector('form').addEventListener('submit', function(event) {
    const password = document.querySelector('input[name="password"]').value;
    const rePassword = document.querySelector('input[name="re_password"]').value;

    if (password !== rePassword) {
        event.preventDefault();
        alert('Passwords do not match. Please try again.');
    }
});