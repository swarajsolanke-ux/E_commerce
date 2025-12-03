// ===== API Configuration =====
const API_BASE = (window.location.origin && window.location.origin !== "null") 
  ? window.location.origin 
  : `${location.protocol}//${location.hostname}:5000`;

let state = {
  messages: [],
  history: [],
  isLoading: false,
  theme: 'light',
  username: 'User',
  email: 'user@example.com',
  attachment: null
};

// ===== Product Normalization Functions =====
function basenameFromPath(p) {
  if (!p || typeof p !== 'string') return '';
  const parts = p.split(/[\\/]+/);
  return parts[parts.length - 1] || '';
}

function normalizeProduct(prod) {
  if (!prod || typeof prod !== 'object') return null;
  const name = prod.title || prod.name || (prod.metadata && (prod.metadata.title || prod.metadata.name)) || 'Unknown Product';
  const costRaw = prod.selling_price || prod.cost || (prod.metadata && (prod.metadata.selling_price || prod.metadata.cost)) || '';
  const costNum = (costRaw && typeof costRaw === 'string') ? costRaw.replace(/[^\d.]/g, '') : costRaw;
  const cost = costNum === '' ? '' : Number(costNum);
  const rating = (prod.product_rating != null) ? Number(prod.product_rating) : (prod.rating != null ? Number(prod.rating) : (prod.metadata && prod.metadata.product_rating ? Number(prod.metadata.product_rating) : null));
  const review = prod.description || prod.review || (prod.metadata && (prod.metadata.description || prod.metadata.review)) || '';

  let image = prod.image || (prod.metadata && (prod.metadata.image_path || (prod.metadata.image_urls && prod.metadata.image_urls[0]))) || null;

  if (image && typeof image === 'string') {
    const trimmed = image.trim();
    if (/^https?:\/\//i.test(trimmed)) {
      image = trimmed;
    } else if (trimmed.startsWith('/images/')) {
      image = API_BASE + trimmed;
    } else {
      const b = basenameFromPath(trimmed);
      image = b ? API_BASE + '/images/' + encodeURIComponent(b) : null;
    }
  } else {
    image = null;
  }

  return { name, cost, rating, review, image, metadata: prod.metadata || {} };
}

function normalizeRecommendation(rec) {
  if (!rec || typeof rec !== 'object') return null;
  const name = rec.title || rec.name || 'Unknown';
  const costRaw = rec.selling_price || rec.cost || '';
  const costNum = (costRaw && typeof costRaw === 'string') ? costRaw.replace(/[^\d.]/g, '') : costRaw;
  const cost = costNum === '' ? '' : Number(costNum);
  const rating = (rec.product_rating != null) ? Number(rec.product_rating) : (rec.rating != null ? Number(rec.rating) : null);

  let image = rec.image || (rec.metadata && (rec.metadata.image_path || (rec.metadata.image_urls && rec.metadata.image_urls[0]))) || null;
  if (image && typeof image === 'string') {
    const t = image.trim();
    if (/^https?:\/\//i.test(t)) {
      image = t;
    } else if (t.startsWith('/images/')) {
      image = API_BASE + t;
    } else {
      const b = basenameFromPath(t);
      image = b ? API_BASE + '/images/' + encodeURIComponent(b) : null;
    }
  } else {
    image = null;
  }

  return { name, cost, rating, image };
}

// image preview functionality
// ===== Image Preview Functions =====
function createImagePreviewModal() {
  const modal = document.createElement('div');
  modal.id = 'imagePreviewModal';
  modal.className = 'image-preview-modal';
  modal.style.cssText = `
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.8);
    z-index: 1000;
    align-items: center;
    justify-content: center;
  `;
  modal.innerHTML = `
    <div class="image-preview-overlay" style="
      width: 100%;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
    " onclick="closeImagePreview()">
      <div class="image-preview-content" style="
        position: relative;
        max-width: 90%;
        max-height: 90%;
      " onclick="event.stopPropagation()">
        <img id="previewImg" src="" alt="Preview" style="
          max-width: 100%;
          max-height: 100%;
          object-fit: contain;
          border-radius: 8px;
        ">
        <button class="close-preview" onclick="closeImagePreview()" style="
          position: absolute;
          top: -10px;
          right: -10px;
          background: white;
          border: none;
          border-radius: 50%;
          width: 30px;
          height: 30px;
          cursor: pointer;
          font-size: 20px;
          font-weight: bold;
          color: black;
          display: flex;
          align-items: center;
          justify-content: center;
        ">&times;</button>
      </div>
    </div>
  `;
  document.body.appendChild(modal);
}

function showImagePreview(src) {
  let modal = document.getElementById('imagePreviewModal');
  if (!modal) {
    createImagePreviewModal();
    modal = document.getElementById('imagePreviewModal');
  }
  const img = document.getElementById('previewImg');
  img.src = src;
  modal.style.display = 'flex';
}

function closeImagePreview() {
  const modal = document.getElementById('imagePreviewModal');
  if (modal) {
    modal.style.display = 'none';
  }
}

function handleImageClick(e) {
  if (e.target.classList.contains('product-preview-img')) {
    const src = e.target.src;
    showImagePreview(src);
  }
}


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
  
  document.addEventListener('click', handleImageClick);
  console.log(`API Base URL: ${API_BASE}`);
  
  // Load chat history from database and then render
  loadChatHistoryFromDB().then(() => {
    renderMessages();
    renderHistoryFromDB();
    
    // Show welcome message if no messages
    if (state.messages.length === 0) {
      setTimeout(() => {
        addMessage({
          role: 'assistant',
          text: "Hello! I'm your AI E-commerce assistant. I can help you find products, check prices, and provide recommendations. What are you looking for today?"
        });
      }, 500);
    }
  });
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
  // Load history from database when drawer opens
  renderHistoryFromDB();
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
  
  // Call the backend API
  fetchQueryResponse(text);
}

// ===== Backend API Integration =====
async function fetchQueryResponse(query) {
  // Get user_id from localStorage
  const userId = localStorage.getItem('user_id');
  if (!userId) {
    state.isLoading = false;
    addMessage({
      role: 'assistant',
      text: 'Please login to use the chatbot.'
    });
    showToast('Authentication Required', 'Please login to continue.');
    // Optionally redirect to login
    // window.location.href = '/login';
    return;
  }

  try {
    const resp = await fetch(`${API_BASE}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query, user_id: parseInt(userId) })
    });

    state.isLoading = false;

    if (!resp.ok) {
      const err = await resp.json().catch(() => null);
      throw new Error((err && err.detail) ? err.detail : `HTTP ${resp.status}`);
    }

    const data = await resp.json();

    // Handle list format response
    if (data.list_format && Array.isArray(data.categories)) {
      addMessage({
        role: 'assistant',
        text: data.response || 'Here is the list:',
        isList: true,
        list: data.categories
      });
    } else {
      // Regular text response
      addMessage({
        role: 'assistant',
        text: data.response || 'Here are the results'
      });
    }

    // Show product card if available
    if (data.main_product || (Array.isArray(data.products) && data.products.length > 0) || (data.products && typeof data.products === 'object')) {
      // Get main product (first product or main_product field)
      const mainProduct = data.main_product || (Array.isArray(data.products) ? data.products[0] : data.products);
      const normalized = normalizeProduct(mainProduct);
      
      // Get recommendations (limit to 3)
      const recommendations = (data.recommendations || []).slice(0, 3).map(normalizeRecommendation).filter(x => x);
      
      if (normalized) {
        addMessage({
          role: 'product',
          product: normalized,
          recommendations: recommendations
        });
      }
    }

    renderMessages();
    renderHistory();

  } catch (e) {
    state.isLoading = false;
    console.error('API Error:', e);
    addMessage({
      role: 'assistant',
      text: `Error: ${e.message}. Please try again.`
    });
    renderMessages();
  }
}

// ===== Load Chat History from Database =====
async function loadChatHistoryFromDB() {
  const userId = localStorage.getItem('user_id');
  if (!userId) return;

  try {
    const resp = await fetch(`${API_BASE}/chat_history/${userId}?limit=100`);
    if (resp.ok) {
      const data = await resp.json();
      if (data.success && data.chat_history && data.chat_history.length > 0) {
        // Convert database history to message format
        const historyMsgs = [];
        data.chat_history.reverse().forEach(item => {
          // Add user query
          historyMsgs.push({
            id: generateId(),
            role: 'user',
            text: item.query,
            timestamp: new Date(item.timestamp).getTime()
          });
          // Add assistant response
          historyMsgs.push({
            id: generateId(),
            role: 'assistant',
            text: item.response,
            timestamp: new Date(item.timestamp).getTime()
          });
          // Add product if available
          if (item.products && Array.isArray(item.products) && item.products.length > 0) {
            const mainProduct = item.products[0];
            const recommendations = item.products.slice(1, 4).map(normalizeRecommendation).filter(x => x);
            historyMsgs.push({
              id: generateId(),
              role: 'product',
              product: normalizeProduct(mainProduct),
              recommendations: recommendations,
              timestamp: new Date(item.timestamp).getTime()
            });
          }
        });
        
        // Merge with existing messages (avoid duplicates)
        const existingTexts = new Set(state.messages.map(m => m.text || (m.product && m.product.name)));
        historyMsgs.forEach(msg => {
          const msgText = msg.text || (msg.product && msg.product.name);
          if (!existingTexts.has(msgText)) {
            state.messages.push(msg);
            existingTexts.add(msgText);
          }
        });
        saveState();
      }
    }
  } catch (e) {
    console.error('Error loading chat history:', e);
  }
}

// ===== Render History from Database =====
async function renderHistoryFromDB() {
  const userId = localStorage.getItem('user_id');
  
  if (!userId) {
    elements.historyList.innerHTML = `
      <div class="history-empty">
        <i data-lucide="clock"></i>
        <p>Please login to view chat history</p>
      </div>
    `;
    lucide.createIcons();
    return;
  }

  elements.historyList.innerHTML = `
    <div class="history-empty">
      <p>Loading history...</p>
    </div>
  `;

  try {
    const resp = await fetch(`${API_BASE}/chat_history/${userId}?limit=50`);
    if (resp.ok) {
      const data = await resp.json();
      
      if (data.success && data.chat_history && data.chat_history.length > 0) {
        elements.historyList.innerHTML = data.chat_history.map((item, idx) => `
          <button class="history-item" onclick="selectHistory('${escapeHtml(item.query).replace(/'/g, "\\'")}'" style="animation-delay: ${idx * 0.05}s">
            <p class="history-query">${escapeHtml(item.query)}</p>
            <p class="history-time">
              <i data-lucide="clock"></i>
              ${formatDate(new Date(item.timestamp).getTime())}
            </p>
            ${item.response ? `<p class="history-response">${escapeHtml(item.response.substring(0, 60))}...</p>` : ''}
          </button>
        `).join('');
      } else {
        elements.historyList.innerHTML = `
          <div class="history-empty">
            <i data-lucide="clock"></i>
            <p>No conversations yet</p>
            <p style="font-size: 0.875rem;">Start chatting to see your history</p>
          </div>
        `;
      }
    } else {
      // Fallback to local history
      renderHistory();
      return;
    }
  } catch (e) {
    console.error('Error loading history:', e);
    // Fallback to local history
    renderHistory();
    return;
  }
  
  lucide.createIcons();
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
async function handleLogout() {
  if (confirm('Are you sure you want to logout?')) {
    closeModal();
    
    // Call logout API
    const userId = localStorage.getItem('user_id');
    if (userId) {
      try {
        await fetch(`${API_BASE}/logout`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: parseInt(userId) })
        });
      } catch (e) {
        console.error('Logout error:', e);
      }
    }
    
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
    
    // Optionally redirect to login
     window.location.href = '/login';
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
