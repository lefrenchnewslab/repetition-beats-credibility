import re
h=open('dist/index.html').read()
clean='<script>(function(){try{var m=localStorage.getItem("rbc.theme");if(m==="light"||m==="dark")document.documentElement.dataset.theme=m;}catch(e){}})();</script>'
h=re.sub(r'<script src="\./theme-preflight\.js"[^>]*></script>',clean,h)
h=re.sub(r'<!doctype html>\s*','',h,flags=re.I)
for p in [r'</?html[^>]*>',r'</?head>',r'</?body[^>]*>',r'<meta charset="utf-8"\s*/?>',r'<meta name="viewport"[^>]*>']: h=re.sub(p,'',h)
t=re.search(r'<title>.*?</title>',h).group(0); h=h.replace(t,''); m=re.search(r'<meta name="claude:viewport-fit-sides"[^>]*>',h).group(0); h=h.replace(m,'')
h=t+'\n'+m+'\n'+h.strip()
assert 'theme-preflight' not in h
for out in ['/mnt/user-data/outputs/viz/repetition-beats-credibility.html','/mnt/user-data/outputs/rbc-site/repetition-beats-credibility.html']: open(out,'w').write(h)
print('ok',len(h))
