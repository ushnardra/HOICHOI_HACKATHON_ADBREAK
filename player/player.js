// Demo player: reads the VMAP, plays the episode, cuts to the ad at each break and resumes at the same frame.
const $ = id => document.getElementById(id);
const content = $('content'), ad = $('ad');
let breaks = [], debug = null, current = null, adPlaying = null, pendingSeek = null;

const fmt = t => { t = Math.max(0, t || 0); return Math.floor(t / 60) + ':' + String(Math.floor(t % 60)).padStart(2, '0'); };
const hms = s => { const [h, m, x] = s.split(':').map(Number); return h * 3600 + m * 60 + x; };

async function loadVideos() {
  const vids = (await (await fetch('/api/videos')).json()).filter(v => v.processed);
  $('pick').innerHTML = vids.map(v => `<option value="${v.name}">${v.name} (${fmt(v.duration_sec)})</option>`).join('');
  $('pick').onchange = () => load($('pick').value);
  const want = new URLSearchParams(location.search).get('v');
  if (vids.length) load(want && vids.some(v => v.name === want) ? want : vids[0].name);
}

async function load(name) {
  current = name; adPlaying = null; pendingSeek = null;
  const [xmlText, dbg] = await Promise.all([fetch(`/vmap/${name}.xml`).then(r => r.text()), fetch(`/debug/${name}.json`).then(r => r.json())]);
  debug = dbg;
  const xml = new DOMParser().parseFromString(xmlText, 'application/xml');
  const NS = 'http://www.iab.net/videosuite/vmap';
  breaks = [...xml.getElementsByTagNameNS(NS, 'AdBreak')].map((b, i) => {
    const p = dbg.placements[i] || {};
    return {
      id: b.getAttribute('breakId'), t: hms(b.getAttribute('timeOffset')),
      title: b.getElementsByTagName('AdTitle')[0].textContent,
      media: b.getElementsByTagName('MediaFile')[0].textContent.trim(),
      why: p.why || '', blocked: p.blocked || [], before: p.scene_before, after: p.scene_after, played: false,
    };
  }).sort((a, b) => a.t - b.t);
  content.src = `/media/${name}.mp4`;
  $('files').innerHTML = `Files: <a href="/vmap/${name}.xml" target="_blank">VMAP</a> · <a href="/debug/${name}.json" target="_blank">debug JSON</a>`;
  content.addEventListener('loadedmetadata', drawBar, { once: true });
  $('pick').value = name;
  drawBar(); drawBreaks(); drawJumps();
}

function drawBar() {
  const bar = $('bar'), sc = debug.scenes || [];
  const T = content.duration || (sc.length ? sc[sc.length - 1].end : 1);  // markers show before the video loads
  bar.querySelectorAll('.mk,.sens').forEach(e => e.remove());
  (debug.scenes || []).filter(s => s.sensitive).forEach(s => {
    const e = document.createElement('div'); e.className = 'sens';
    e.style.left = (s.start / T * 100) + '%'; e.style.width = ((s.end - s.start) / T * 100) + '%'; bar.appendChild(e);
  });
  breaks.forEach((b, i) => {
    const m = document.createElement('div'); m.className = 'mk' + (b.played ? ' done' : '');
    m.style.left = (b.t / T * 100) + '%'; m.innerHTML = `<span>AD ${i + 1} · ${fmt(b.t)}</span><i></i>`;
    b.el = m; bar.appendChild(m);
  });
  const left = breaks.filter(b => !b.played).length;
  $('count').innerHTML = breaks.length ? `<b>${breaks.length}</b> ad break${breaks.length > 1 ? 's' : ''} · ${left} still to play` : 'no ad breaks';
}

function drawBreaks() {
  if (!breaks.length && debug) {
    const a = debug.whether.allowed || {}, s = debug.summary || {};
    const why = a.final === 0
      ? (a.by_ad_load === 0 ? 'The video is too short for even one ad within the ad-load limit (one 30 s ad needs about 5 min of video at 10%).'
                            : 'The pacing policy allows no break for a video of this length.')
      : `No scene change passed all the checks: ${s.safe_break_points} of ${s.scene_changes} scene changes had a natural pause, and none of them passed the quality and pacing rules.`;
    $('breaks').innerHTML = `<div class="bk"><div class="body"><div class="title">No ad break in this video</div><div class="why">${why}</div>
      <div class="why">Every decision is in the debug JSON below.</div></div></div>`;
    return;
  }
  $('breaks').innerHTML = breaks.map((b, i) => `<div class="bk${b.played ? ' done' : ''}"><span class="t">${fmt(b.t)}</span>
    <div class="body"><div class="title">Break ${i + 1} · ${b.title}</div><div class="why">${b.why}</div>
    ${b.blocked.length ? `<div class="blk" title="${b.blocked.map(x => x.name).join(', ')}">${b.blocked.length} brand${b.blocked.length > 1 ? 's' : ''} blocked by scene safety (hover for names)</div>` : ''}
    <button data-i="${i}">▶ Watch from 8 s before</button></div></div>`).join('');
  bindJumps($('breaks'));
}

function drawJumps() {}

function bindJumps(root) {
  root.querySelectorAll('button[data-i]').forEach(btn => btn.onclick = () => {
    const b = breaks[+btn.dataset.i]; b.played = false; drawBar(); drawBreaks();
    content.currentTime = Math.max(0, b.t - 8); content.play();
  });
}

function startAd(b, resumeAt) {
  adPlaying = { b, resumeAt };
  content.pause();
  content.currentTime = b.t;                    // exact frame where the ad starts
  $('adTitle').textContent = b.title;
  $('adWhy').innerHTML = `<b>Why here:</b> ${b.why}`;
  $('soon').style.display = 'none';
  ['adbar', 'adWhy', 'adprog'].forEach(id => $(id).style.display = id === 'adbar' ? 'flex' : 'block');
  ad.style.display = 'block';
  ad.src = b.media; ad.currentTime = 0; ad.play();
}

function endAd() {
  if (!adPlaying) return;
  const { b, resumeAt } = adPlaying; adPlaying = null;
  b.played = true; ad.pause(); ad.removeAttribute('src'); ad.load();
  ad.style.display = 'none'; ['adbar', 'adWhy', 'adprog'].forEach(id => $(id).style.display = 'none');
  content.currentTime = resumeAt ?? b.t;       // resume at the same frame (or where the viewer seeked to)
  content.play(); drawBar(); drawBreaks();
}

ad.addEventListener('ended', endAd);
ad.addEventListener('timeupdate', () => {
  $('adLeft').textContent = fmt((ad.duration || 0) - ad.currentTime) + ' left';
  $('adfill').style.width = (ad.currentTime / (ad.duration || 1) * 100) + '%';
});
$('skip').onclick = endAd;

// Check ~60 times a second (timeupdate alone is only ~4 per second, which could overshoot a break by 0.25 s).
function tick() {
  if (!adPlaying && !content.paused) {
    const t = content.currentTime;
    const b = breaks.find(x => !x.played && t >= x.t && t - x.t < 1.0);
    if (b) startAd(b, null);
  }
  // "Ad break in N" chip and a pulsing marker in the last 5 s before a break
  const up = !adPlaying && breaks.find(x => !x.played && x.t > content.currentTime && x.t - content.currentTime <= 5);
  $('soon').style.display = up && !content.paused ? 'block' : 'none';
  if (up) $('soonN').textContent = Math.ceil(up.t - content.currentTime);
  breaks.forEach(x => x.el && x.el.classList.toggle('soon', x === up));
  const T = content.duration || 1;
  $('fill').style.width = (content.currentTime / T * 100) + '%';
  $('time').textContent = fmt(content.currentTime) + ' / ' + fmt(content.duration);
  updatePanel();
  requestAnimationFrame(tick);
}

// Seeking over an unplayed break plays that break first, then continues where the viewer wanted to go.
$('bar').onclick = e => {
  const r = $('bar').getBoundingClientRect(), target = (e.clientX - r.left) / r.width * content.duration;
  const skipped = breaks.filter(b => !b.played && b.t > content.currentTime && b.t < target).pop();
  if (skipped) { startAd(skipped, target); } else { content.currentTime = target; }
};

$('play').onclick = () => { if (adPlaying) return; content.paused ? content.play() : content.pause(); };
content.addEventListener('play', () => $('play').textContent = '❚❚ Pause');
content.addEventListener('pause', () => $('play').textContent = '▶ Play');

function updatePanel() {
  if (!debug) return;
  const t = content.currentTime;
  const s = debug.scenes.find(x => x.start <= t && t < x.end);
  if (s) $('scene').innerHTML = `<span class="pill ${s.sensitive ? 's' : 'o'}">${s.sensitive ? 'SENSITIVE' : 'SAFE'}</span>
      scene ${fmt(s.start)}–${fmt(s.end)}<br>activity <b>${s.activity}</b> · ${s.setting} · ${s.mood}
      ${s.sensitive ? `<div class="muted small">${s.sensitive_reasons.join(' · ')}</div>` : ''}`;
  const n = breaks.find(b => b.t > t - 0.05 && !b.played);
  $('next').innerHTML = adPlaying ? `<b>▶ Ad playing:</b> ${adPlaying.b.title}`
    : n ? `<b>${n.title}</b> in <b>${fmt(n.t - t)}</b> <span class="muted">(at ${fmt(n.t)})</span>` : '<span class="muted">No more ad breaks</span>';
}

loadVideos();
requestAnimationFrame(tick);
