(() => {
  const API_BASE = (window.location.origin && window.location.origin !== "null") 
    ? window.location.origin 
    : `${location.protocol}//${location.hostname}:5000`;
  
  const $ = id => document.getElementById(id);
  const esc = s => String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#039;');

  const stream = $('stream'), input = $('composerInput'), send = $('sendBtn'),
        attach = $('attachInput'), attachLabel = $('attachmentLabel'),
        webBtn = $('webSearchBtn'), drawer = $('leftDrawer'), overlay = $('overlay'),
        openDrawer = $('openDrawer'), closeDrawer = $('closeDrawer'),
        historyList = $('historyList'), clearHist = $('clearHistory'),
        themeToggle = $('themeToggle');

  let msgs = [], attachment = null, currentTheme = 'light';

  const loadTheme = () => {
    try {
      const saved = localStorage.getItem('theme');
      if (saved) {
        currentTheme = saved;
        document.documentElement.setAttribute('data-theme', currentTheme);
        themeToggle.textContent = currentTheme === 'dark' ? '☀️' : '🌙'; 
      }
    } catch(e){}
  };
  const toggleTheme = () => {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', currentTheme);
    themeToggle.textContent = currentTheme === 'dark' ? '☀️' : '🌙';
    try { localStorage.setItem('theme', currentTheme); } catch(e){}
  };

  const save = () => { try { localStorage.setItem('chat_messages', JSON.stringify(msgs)); } catch(e){} };
  const load = () => { try { const s = localStorage.getItem('chat_messages'); if (s) msgs = JSON.parse(s); } catch(e){} };

  function basenameFromPath(p) {
    if (!p || typeof p !== 'string') return '';
    const parts = p.split(/[\\/]+/);
    return parts[parts.length-1] || '';
  }

  function normalizeProduct(prod) {
    if (!prod || typeof prod !== 'object') return null;
    const name = prod.title || prod.name || (prod.metadata && (prod.metadata.title || prod.metadata.name)) || 'Unknown Product';
    const costRaw = prod.selling_price || prod.cost || (prod.metadata && (prod.metadata.selling_price || prod.metadata.cost)) || '';
    const costNum = (costRaw && typeof costRaw === 'string') ? costRaw.replace(/[^\d.]/g,'') : costRaw;
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
    const costNum = (costRaw && typeof costRaw === 'string') ? costRaw.replace(/[^\d.]/g,'') : costRaw;
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

  const render = () => {
    if (msgs.length === 0) {
      stream.innerHTML = `
        <div class="empty-state">
          <div class="icon"></div>
          <h3>Welcome to E-commerce Assistant!</h3>
          <p>Ask me anything about products, prices, or recommendations</p>
        </div>`;
      return;
    }

    stream.innerHTML = '';
    msgs.forEach((m, idx) => {
      const el = document.createElement('div');
      el.style.animationDelay = `${idx * 0.03}s`;

      if (m.role === 'user') {
        el.className = 'msg user';
        el.innerHTML = `
          <div>
            <div class="text">${esc(m.text)}</div>
            <div class="meta">${m.time ? new Date(m.time).toLocaleTimeString() : ''} ${m.attachment ? `<span class="attachment-badge">${esc(m.attachment)}</span>`:''}</div>
          </div>
          <div class="msg-avatar">👤</div>
        `;
      } else if (m.role === 'product') {
        const p = normalizeProduct(m.product || {});
        const recsRaw = m.recommendations || [];
        const recs = Array.isArray(recsRaw) ? recsRaw.map(normalizeRecommendation).filter(x => x) : [];
        const name = esc(p.name || 'Unknown Product');
        const price = (p.cost !== '' && p.cost != null) ? '₹' + Number(p.cost).toFixed(2) : '—';
        const rating = (p.rating != null) ? 'Star ' + Number(p.rating).toFixed(1) : '—';
        const review = esc(p.review || 'No reviews available');
        const img = p.image ? `<img src="${p.image}" alt="${name}" />` : '<div style="font-size:48px;">Package</div>';

        let recHTML = '';
        if (recs.length) {
          recHTML = `<div class="recommendations"><h4>You might also like</h4><div class="rec-grid">`;
          recs.forEach(r => {
            const rn = esc(r.name || 'Unknown');
            const rp = (r.cost !== '' && r.cost != null) ? '₹' + Number(r.cost).toFixed(2) : '—';
            const rr = (r.rating != null) ? 'Star ' + Number(r.rating).toFixed(1) : '—';
            const ri = r.image ? `<img src="${r.image}" alt="${rn}"/>` : '';
            recHTML += `
              <div class="rec-card">
                ${ri || '<div style="height:100px;display:flex;align-items:center;justify-content:center;font-size:32px;">Package</div>'}
                <div class="name">${rn}</div>
                <div class="price">${rp}</div>
                <div class="rating">${rr}</div>
              </div>`;
          });
          recHTML += `</div></div>`;
        }

        el.className = 'msg product ai';
        el.innerHTML = `
          <div class="msg-avatar">🤖</div>
          <div style="max-width:85%;">
            <div class="product-card">
              <div class="thumb">${img}</div>
              <div class="prod-meta">
                <h4 class="prod-title">${name}</h4>
                <div class="prod-price">${price} • ${rating}</div>
                <div class="prod-review">${review}</div>
              </div>
            </div>
            ${recHTML}
            <div class="meta">${m.time ? new Date(m.time).toLocaleTimeString() : ''}</div>
          </div>`;
      } else {
        el.className = 'msg ai';
        const isLoading = m.text === '...';
        const textContent = isLoading ? '<div class="loading-dots"><span></span><span></span><span></span></div>' : esc(m.text);
        el.innerHTML = `
          <div class="msg-avatar">🤖</div>
          <div>
            <div class="text">${textContent}</div>
            <div class="meta">${m.time && !isLoading ? new Date(m.time).toLocaleTimeString() : ''}</div>
          </div>`;
      }
      stream.appendChild(el);
    });
    stream.scrollTop = stream.scrollHeight;
  };

  const renderHist = () => {
    historyList.innerHTML = '';
    const recent = msgs.filter(m => m.role === 'user').slice(-50).reverse();
    if (!recent.length) {
      historyList.innerHTML = `<div class="empty-state" style="padding:40px 20px;"><div class="icon" style="font-size:48px;"></div><p style="color:var(--text-secondary);">No conversations yet</p></div>`;
      return;
    }
    recent.forEach(q => {
      const it = document.createElement('div');
      it.className = 'history-item';
      it.innerHTML = `<div class="q">${esc(q.text)}</div><div class="t">${new Date(q.time).toLocaleString()}</div>`;
      it.onclick = () => { input.value = q.text; input.focus(); drawer.classList.add('hidden'); overlay.classList.add('hidden'); };
      historyList.appendChild(it);
    });
  };

  const sendQuery = async () => {
    let txt = input.value.trim();
    if (!txt && !attachment) return;
    if (attachment) txt += ` [attachment: ${attachment.name}]`;

    msgs.push({ role: 'user', text: txt, time: Date.now(), attachment: attachment?.name || null });
    save(); render(); renderHist();

    input.value = ''; attachment = null; attachLabel.innerHTML = '';

    msgs.push({ role: 'assistant', text: '...', time: Date.now() });
    save(); render();

    try {
      const resp = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: txt })
      });

      if (msgs[msgs.length - 1].role === 'assistant' && msgs[msgs.length - 1].text === '...') msgs.pop();

      if (!resp.ok) {
        const err = await resp.json().catch(()=>null);
        throw new Error((err && err.detail) ? err.detail : `HTTP ${resp.status}`);
      }

      const data = await resp.json();

      msgs.push({ role: 'assistant', text: data.response || 'Here are the results', time: Date.now() });

      if (Array.isArray(data.products) && data.products.length) {
        data.products.forEach(praw => {
          const normalized = normalizeProduct(praw);
          msgs.push({ role: 'product', product: normalized, recommendations: (data.recommendations||[]), time: Date.now() });
        });
      } else if (data.products && typeof data.products === 'object') {
        const normalized = normalizeProduct(data.products);
        msgs.push({ role: 'product', product: normalized, recommendations: (data.recommendations||[]), time: Date.now() });
      }

      save(); render(); renderHist();
    } catch (e) {
      if (msgs[msgs.length - 1].role === 'assistant' && msgs[msgs.length - 1].text === '...') msgs.pop();
      msgs.push({ role: 'assistant', text: `Error: ${e.message}. Please try again.`, time: Date.now() });
      save(); render(); renderHist();
    }
  };

  const autoResize = () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
  };

  attach.addEventListener('change', e => {
    const f = e.target.files?.[0];
    if (!f) return;
    attachment = f;
    attachLabel.innerHTML = `<span class="attachment-badge">Attached ${esc(f.name)}</span>`;
  });
  input.addEventListener('input', autoResize);
  input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuery(); } });
  send.addEventListener('click', sendQuery);
  webBtn.addEventListener('click', () => { const q = input.value.trim(); if (q) window.open(`https://www.google.com/search?q=${encodeURIComponent(q)}`, '_blank'); });
  openDrawer.addEventListener('click', () => { drawer.classList.remove('hidden'); overlay.classList.remove('hidden'); renderHist(); });
  closeDrawer.addEventListener('click', () => { drawer.classList.add('hidden'); overlay.classList.add('hidden'); });
  overlay.addEventListener('click', () => { drawer.classList.add('hidden'); overlay.classList.add('hidden'); });
  clearHist.addEventListener('click', () => { if (confirm('Clear all chat history? This cannot be undone.')) { msgs = []; save(); render(); renderHist(); } });
  themeToggle.addEventListener('click', toggleTheme);

  console.log(`API Base URL:$(API_BASE)`);
  loadTheme();
  load(); render(); renderHist();
  if (msgs.length === 0) {
    setTimeout(() => {
      msgs.push({ role: 'assistant', text: "Hello! I'm your AI Ecommerce assistant. I can help you find products, check prices, and provide recommendations. What are you looking for today?", time: Date.now() });
      save(); render();
    }, 500);
  }
})();