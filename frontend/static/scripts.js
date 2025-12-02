let state = {
  messages: [],
  history: [],
  isLoading: false,
  theme: 'light',
  username: 'User',
  email: 'user@example.com',
  attachment: null
};

// ===== DOM Elements =====
const elements = {
  app: document.getElementById('app'),
  overlay: document.getElementById('overlay'),
  chatDrawer: document.getElementById('chatDrawer'),
  historyList: document.getElementById('historyList'),
  accountModal: document.getElementById('accountModal'),
  messagesContainer: document.getElementById('messagesContainer'),
  emptyState: document.getElementById('emptyState'),
  messageInput: document.getElementById('messageInput'),
  sendBtn: document.getElementById('sendBtn'),
  attachBtn: document.getElementById('attachBtn'),
  fileInput: document.getElementById('fileInput'),
  attachmentIndicator: document.getElementById('attachmentIndicator'),
  attachmentName: document.getElementById('attachmentName'),
  removeAttachment: document.getElementById('removeAttachment'),
  toastContainer: document.getElementById('toastContainer'),
  menuBtn: document.getElementById('menuBtn'),
  closeDrawer: document.getElementById('closeDrawer'),
  closeDrawerBtn: document.getElementById('closeDrawerBtn'),
  clearHistoryBtn: document.getElementById('clearHistoryBtn'),
  newChatBtn: document.getElementById('newChatBtn'),
  themeBtn: document.getElementById('themeBtn'),
  themeIcon: document.getElementById('themeIcon'),
  accountBtn: document.getElementById('accountBtn'),
  closeModal: document.getElementById('closeModal'),
  webSearchBtn: document.getElementById('webSearchBtn'),
  userAvatar: document.getElementById('userAvatar'),
  userName: document.getElementById('userName'),
  userEmail: document.getElementById('userEmail'),
  accountInitial: document.getElementById('accountInitial')
};

// ===== Initialize =====
function init() {
  loadState();
  applyTheme();
  updateUserDisplay();
  renderHistory();
  renderMessages();
  setupEventListeners();
  lucide.createIcons();
  
  // Show welcome message if no messages
  if (state.messages.length === 0) {
    setTimeout(() => {
      addMessage({
        role: 'assistant',
        text: "Hello! I'm your AI E-commerce assistant. I can help you find products, check prices, and provide recommendations. What are you looking for today?"
      });
    }, 500);
  }
}

// ===== State Persistence =====
function loadState() {
  const savedMessages = localStorage.getItem('chat_messages');
  const savedHistory = localStorage.getItem('chat_history');
  const savedTheme = localStorage.getItem('theme');
  const savedUsername = localStorage.getItem('username');
  const savedEmail = localStorage.getItem('email');
  
  if (savedMessages) state.messages = JSON.parse(savedMessages);
  if (savedHistory) state.history = JSON.parse(savedHistory);
  if (savedTheme) state.theme = savedTheme;
  if (savedUsername) state.username = savedUsername;
  if (savedEmail) state.email = savedEmail;
}

function saveState() {
  localStorage.setItem('chat_messages', JSON.stringify(state.messages));
  localStorage.setItem('chat_history', JSON.stringify(state.history));
  localStorage.setItem('theme', state.theme);
}

// ===== Theme =====
function applyTheme() {
  if (state.theme === 'dark') {
    elements.app.classList.add('dark');
    elements.themeIcon.setAttribute('data-lucide', 'moon');
  } else {
    elements.app.classList.remove('dark');
    elements.themeIcon.setAttribute('data-lucide', 'sun');
  }
  lucide.createIcons();
}

function toggleTheme() {
  state.theme = state.theme === 'light' ? 'dark' : 'light';
  applyTheme();
  saveState();
}

// ===== User Display =====
function updateUserDisplay() {
  const initial = state.username.charAt(0).toUpperCase();
  elements.userAvatar.textContent = initial;
  elements.userName.textContent = state.username;
  elements.userEmail.textContent = state.email;
  elements.accountInitial.textContent = initial;
}

// ===== Event Listeners =====
function setupEventListeners() {
  // Drawer
  elements.menuBtn.addEventListener('click', openDrawer);
  elements.closeDrawer.addEventListener('click', closeDrawer);
  elements.closeDrawerBtn.addEventListener('click', closeDrawer);
  elements.overlay.addEventListener('click', closeDrawer);
  elements.clearHistoryBtn.addEventListener('click', clearHistory);
  
  // Modal
  elements.accountBtn.addEventListener('click', openModal);
  elements.closeModal.addEventListener('click', closeModal);
  elements.accountModal.addEventListener('click', (e) => {
    if (e.target === elements.accountModal) closeModal();
  });
  
  // Theme
  elements.themeBtn.addEventListener('click', toggleTheme);
  
  // New Chat
  elements.newChatBtn.addEventListener('click', newChat);
  
  // Message Input
  elements.messageInput.addEventListener('input', handleInputChange);
  elements.messageInput.addEventListener('keydown', handleKeyDown);
  elements.sendBtn.addEventListener('click', sendMessage);
  
  // Attachment
  elements.attachBtn.addEventListener('click', () => elements.fileInput.click());
  elements.fileInput.addEventListener('change', handleFileSelect);
  elements.removeAttachment.addEventListener('click', removeAttachment);
  
  // Web Search
  elements.webSearchBtn.addEventListener('click', webSearch);
}

// ===== Drawer =====
function openDrawer() {
  elements.chatDrawer.classList.add('open');
  elements.overlay.classList.add('active');
}

function closeDrawer() {
  elements.chatDrawer.classList.remove('open');
  elements.overlay.classList.remove('active');
}

// ===== Modal =====
function openModal() {
  elements.accountModal.classList.add('active');
}

function closeModal() {
  elements.accountModal.classList.remove('active');
}

// ===== Messages =====
function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

function addMessage(msg) {
  const message = {
    id: generateId(),
    timestamp: Date.now(),
    ...msg
  };
  state.messages.push(message);
  saveState();
  renderMessages();
  scrollToBottom();
  
  // Add to history if user message
  if (msg.role === 'user') {
    addToHistory(msg.text);
  }
}

function addToHistory(query) {
  const historyItem = {
    id: generateId(),
    query: query,
    timestamp: Date.now(),
    response: ''
  };
  
  // Update with last assistant response
  setTimeout(() => {
    const lastAssistant = state.messages.filter(m => m.role === 'assistant').pop();
    if (lastAssistant) {
      historyItem.response = lastAssistant.text;
    }
    state.history.unshift(historyItem);
    if (state.history.length > 50) state.history.pop();
    saveState();
    renderHistory();
  }, 2000);
}

function clearMessages() {
  state.messages = [];
  saveState();
  renderMessages();
}

function clearHistory() {
  if (confirm('Clear all chat history? This cannot be undone.')) {
    state.history = [];
    saveState();
    renderHistory();
    showToast('Cleared', 'Chat history has been cleared.');
  }
}

function newChat() {
  clearMessages();
  setTimeout(() => {
    addMessage({
      role: 'assistant',
      text: "Hello! I'm your AI E-commerce assistant. I can help you find products, check prices, and provide recommendations. What are you looking for today?"
    });
  }, 300);
}

// ===== Render Messages =====
function renderMessages() {
  // Hide empty state if there are messages
  if (state.messages.length > 0) {
    elements.emptyState.style.display = 'none';
  } else {
    elements.emptyState.style.display = 'flex';
  }
  
  // Remove all message elements (keep empty state)
  const existingMessages = elements.messagesContainer.querySelectorAll('.message');
  existingMessages.forEach(el => el.remove());
  
  // Render messages
  state.messages.forEach((msg, idx) => {
    const messageEl = createMessageElement(msg, idx);
    elements.messagesContainer.appendChild(messageEl);
  });
  
  // Add loading indicator if loading
  if (state.isLoading) {
    const loadingEl = createLoadingElement();
    elements.messagesContainer.appendChild(loadingEl);
  }
  
  lucide.createIcons();
}

function createMessageElement(msg, idx) {
  const div = document.createElement('div');
  div.className = `message ${msg.role === 'user' ? 'message-user' : ''}`;
  div.style.animationDelay = `${idx * 0.03}s`;
  
  if (msg.role === 'product' && msg.product) {
    div.innerHTML = createProductCardHTML(msg);
  } else if (msg.role === 'user') {
    div.innerHTML = createUserMessageHTML(msg);
  } else {
    div.innerHTML = createAssistantMessageHTML(msg);
  }
  
  return div;
}

function createAssistantMessageHTML(msg) {
  return `
    <div class="message-avatar message-avatar-bot">
      <i data-lucide="bot"></i>
    </div>
    <div class="message-content">
      <div class="message-bubble message-bubble-bot">
        <p class="message-text">${escapeHtml(msg.text)}</p>
      </div>
      <p class="message-time">${formatTime(msg.timestamp)}</p>
    </div>
  `;
}

function createUserMessageHTML(msg) {
  const attachmentHTML = msg.attachment ? `
    <span class="message-attachment">
      <i data-lucide="paperclip"></i>
      ${escapeHtml(msg.attachment)}
    </span>
  ` : '';
  
  return `
    <div class="message-content">
      <div class="message-bubble message-bubble-user">
        <p class="message-text">${escapeHtml(msg.text)}</p>
      </div>
      <div style="display: flex; align-items: center; justify-content: flex-end; gap: 0.5rem; margin-top: 0.375rem;">
        <p class="message-time">${formatTime(msg.timestamp)}</p>
        ${attachmentHTML}
      </div>
    </div>
    <div class="message-avatar message-avatar-user">
      <i data-lucide="user"></i>
    </div>
  `;
}

function createProductCardHTML(msg) {
  const product = msg.product;
  const recommendations = msg.recommendations || [];
  
  let recHTML = '';
  if (recommendations.length > 0) {
    recHTML = `
      <div class="recommendations">
        <h4 class="recommendations-title">
          <i data-lucide="sparkles"></i>
          You might also like
        </h4>
        <div class="recommendations-grid">
          ${recommendations.map(rec => `
            <div class="rec-card">
              <div class="rec-image">
                ${rec.image ? `<img src="${rec.image}" alt="${escapeHtml(rec.name)}">` : '<i data-lucide="package"></i>'}
              </div>
              <p class="rec-name">${escapeHtml(rec.name)}</p>
              <p class="rec-price">${formatPrice(rec.cost)}</p>
              ${rec.rating ? `
                <p class="rec-rating">
                  <i data-lucide="star"></i>
                  ${rec.rating.toFixed(1)}
                </p>
              ` : ''}
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }
  
  return `
    <div class="message-avatar message-avatar-bot">
      <i data-lucide="bot"></i>
    </div>
    <div class="message-content" style="max-width: 85%;">
      <div class="product-card">
        <div class="product-main">
          <div class="product-image">
            ${product.image ? `<img src="${product.image}" alt="${escapeHtml(product.name)}">` : '<i data-lucide="package"></i>'}
          </div>
          <div class="product-info">
            <h4 class="product-name">${escapeHtml(product.name)}</h4>
            <div class="product-meta">
              <span class="product-price">${formatPrice(product.cost)}</span>
              ${product.rating ? `
                <span class="product-rating">
                  <i data-lucide="star"></i>
                  ${product.rating.toFixed(1)}
                </span>
              ` : ''}
            </div>
            <p class="product-review">${product.review || 'No reviews available'}</p>
          </div>
        </div>
        ${recHTML}
      </div>
      <p class="message-time">${formatTime(msg.timestamp)}</p>
    </div>
  `;
}

function createLoadingElement() {
  const div = document.createElement('div');
  div.className = 'message';
  div.innerHTML = `
    <div class="message-avatar message-avatar-bot">
      <i data-lucide="bot"></i>
    </div>
    <div class="message-content">
      <div class="message-bubble message-bubble-bot">
        <div class="loading-dots">
          <span class="loading-dot"></span>
          <span class="loading-dot"></span>
          <span class="loading-dot"></span>
        </div>
      </div>
    </div>
  `;
  return div;
}

// ===== Render History =====
function renderHistory() {
  if (state.history.length === 0) {
    elements.historyList.innerHTML = `
      <div class="history-empty">
        <i data-lucide="clock"></i>
        <p>No conversations yet</p>
        <p style="font-size: 0.875rem;">Start chatting to see your history</p>
      </div>
    `;
  } else {
    elements.historyList.innerHTML = state.history.map((item, idx) => `
      <button class="history-item" onclick="selectHistory('${escapeHtml(item.query)}')" style="animation-delay: ${idx * 0.05}s">
        <p class="history-query">${escapeHtml(item.query)}</p>
        <p class="history-time">
          <i data-lucide="clock"></i>
          ${formatDate(item.timestamp)}
        </p>
        ${item.response ? `<p class="history-response">${escapeHtml(item.response.substring(0, 60))}...</p>` : ''}
      </button>
    `).join('');
  }
  lucide.createIcons();
}

function selectHistory(query) {
  elements.messageInput.value = query;
  elements.messageInput.focus();
  closeDrawer();
  updateSendButton();
}

// ===== Input Handling =====
function handleInputChange() {
  // Auto-resize textarea
  elements.messageInput.style.height = 'auto';
  elements.messageInput.style.height = Math.min(elements.messageInput.scrollHeight, 120) + 'px';
  updateSendButton();
}

function handleKeyDown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function updateSendButton() {
  const hasContent = elements.messageInput.value.trim() || state.attachment;
  elements.sendBtn.disabled = !hasContent || state.isLoading;
}

// ===== File Handling =====
function handleFileSelect(e) {
  const file = e.target.files?.[0];
  if (file) {
    state.attachment = file;
    elements.attachmentName.textContent = file.name;
    elements.attachmentIndicator.style.display = 'flex';
    updateSendButton();
  }
}

function removeAttachment() {
  state.attachment = null;
  elements.fileInput.value = '';
  elements.attachmentIndicator.style.display = 'none';
  updateSendButton();
}

// ===== Send Message =====
function sendMessage() {
  const text = elements.messageInput.value.trim();
  if (!text && !state.attachment) return;
  if (state.isLoading) return;
  
  const messageText = state.attachment ? `${text} [attachment: ${state.attachment.name}]` : text;
  
  // Add user message
  addMessage({
    role: 'user',
    text: messageText,
    attachment: state.attachment?.name
  });
  
  // Clear input
  elements.messageInput.value = '';
  elements.messageInput.style.height = 'auto';
  removeAttachment();
  updateSendButton();
  
  // Show loading and get response
  state.isLoading = true;
  renderMessages();
  
  setTimeout(() => {
    simulateResponse(text);
  }, 1500);
}

function simulateResponse(query) {
  state.isLoading = false;
  
  const responses = [
    {
      text: "I found some great options for you! Here are my top recommendations based on your search.",
      hasProduct: true
    },
    {
      text: "Based on your preferences, I recommend checking out these products. They offer excellent value for money and have great customer reviews.",
      hasProduct: true
    },
    {
      text: "Here's what I found! These items match your criteria and are currently available with special discounts.",
      hasProduct: false
    }
  ];
  
  const response = responses[Math.floor(Math.random() * responses.length)];
  
  addMessage({
    role: 'assistant',
    text: response.text
  });
  
  if (response.hasProduct) {
    setTimeout(() => {
      addMessage({
        role: 'product',
        product: {
          name: 'Premium Wireless Headphones',
          cost: 4999.99,
          rating: 4.5,
          review: 'Excellent sound quality with active noise cancellation. Battery lasts up to 30 hours. Comfortable for extended use.',
          image: null
        },
        recommendations: [
          { name: 'Budget Earbuds', cost: 1299, rating: 4.2, review: '', image: null },
          { name: 'Gaming Headset Pro', cost: 3499, rating: 4.7, review: '', image: null },
          { name: 'Sports Wireless Buds', cost: 2199, rating: 4.4, review: '', image: null }
        ]
      });
    }, 500);
  }
}

// ===== Web Search =====
function webSearch() {
  const text = elements.messageInput.value.trim();
  if (text) {
    window.open(`https://www.google.com/search?q=${encodeURIComponent(text)}`, '_blank');
  }
}

// ===== Suggestions =====
function sendSuggestion(text) {
  elements.messageInput.value = text;
  elements.messageInput.focus();
  updateSendButton();
}

// ===== Toast =====
function showToast(title, description) {
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.innerHTML = `
    <p class="toast-title">${escapeHtml(title)}</p>
    <p class="toast-description">${escapeHtml(description)}</p>
  `;
  elements.toastContainer.appendChild(toast);
  
  setTimeout(() => {
    toast.remove();
  }, 3000);
}

// ===== Logout =====
function handleLogout() {
  if (confirm('Are you sure you want to logout?')) {
    closeModal();
    localStorage.removeItem('user_id');
    localStorage.removeItem('email');
    localStorage.removeItem('username');
    localStorage.removeItem('chat_messages');
    localStorage.removeItem('chat_history');
    showToast('Logged out', 'You have been logged out successfully.');
    
    // Reset state
    state.messages = [];
    state.history = [];
    renderMessages();
    renderHistory();
  }
}

// ===== Utilities =====
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function formatTime(timestamp) {
  return new Date(timestamp).toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit'
  });
}

function formatDate(timestamp) {
  return new Date(timestamp).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
}

function formatPrice(cost) {
  if (cost === '' || cost === null || cost === undefined) return '—';
  const num = typeof cost === 'string' ? parseFloat(cost.replace(/[^\d.]/g, '')) : cost;
  return isNaN(num) ? '—' : `₹${num.toFixed(2)}`;
}

function scrollToBottom() {
  setTimeout(() => {
    const chatArea = document.getElementById('chatArea');
    chatArea.scrollTop = chatArea.scrollHeight;
  }, 100);
}

// ===== Initialize on DOM Ready =====
document.addEventListener('DOMContentLoaded', init);
