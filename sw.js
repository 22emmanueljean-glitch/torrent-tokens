self.addEventListener("install", (e)=> self.skipWaiting());
self.addEventListener("activate", (e)=> self.clients.claim());
self.addEventListener("fetch", ()=>{}); // network-first; minimal SW
