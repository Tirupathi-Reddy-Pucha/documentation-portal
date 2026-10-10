/**
 * 🔐 AUTHENTICATION MODULE
 * Manages JWT (JSON Web Token) storage and route guarding.
 * Security Note for FYP: In a production enterprise app, we'd use HTTP-only cookies to prevent XSS. 
 * For this prototype, localStorage is used for simplicity and statelessness.
 */
export function getToken() {
  return localStorage.getItem('access_token') || '';
}

export function getRole() {
  return localStorage.getItem('user_role');
}

export function getUsername() {
  return localStorage.getItem('username');
}

export function setAuth(token, role, username) {
  localStorage.setItem('access_token', token);
  localStorage.setItem('user_role', role);
  localStorage.setItem('username', username);
}

export function logout() {
  localStorage.removeItem('access_token');
  localStorage.removeItem('user_role');
  localStorage.removeItem('username');
  window.location.href = 'index.html';
}

export function requireRole(expectedRole) {
  const token = getToken();
  const role = getRole();
  if (!token || role !== expectedRole) {
    window.location.href = 'index.html';
  }
}
