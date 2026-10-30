/**
 * Anonymous, opt-in Umami integration for Folirina Web.
 * No artwork, filenames, document content, paths, counts, user IDs or pixels
 * are ever passed to track(). Only pre-approved constant event names are sent.
 * Pageviews are emitted automatically by the provider script once configured.
 * The tracker is idle (and makes no network requests) without a Website ID.
 */
import {UMAMI_WEBSITE_ID} from './analytics-config.js';

export const APPROVED_EVENTS = Object.freeze([
  'source_import_success',
  'target_import_success',
  'workspace_open_success',
  'review_apply_success',
  'page_export_success',
  'book_export_success',
  'mask_export_success',
  'workspace_save_success',
]);
const permitted = new Set(APPROVED_EVENTS);
const isValidId = id => typeof id === 'string' && /^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$/i.test(id);

// Injection makes the tracker testable without a network connection.
export function createAnonymousTracker({websiteId=UMAMI_WEBSITE_ID, win=typeof window==='undefined'?null:window,
    doc=typeof document==='undefined'?null:document, host='amster-ilvil.github.io'}={}){
  let started=false;
  let ready=false;
  const pending=[];
  const privacyOptOut=()=>!win || win.navigator?.globalPrivacyControl===true ||
      win.navigator?.doNotTrack==='1' || win.navigator?.msDoNotTrack==='1';
  const active=()=>Boolean(doc && win && win.location?.hostname===host && isValidId(websiteId) && !privacyOptOut());
  const flush=()=>{
    if(!active() || typeof win.umami?.track!=='function')return;
    ready=true;
    while(pending.length){
      const event=pending.shift();
      try{win.umami.track(event)}catch(_e){/* Analytics must never interrupt editing. */}
    }
  };
  function initialize(){
    if(started || !active())return false;
    started=true;
    try{
      const script=doc.createElement('script');
      script.src='https://cloud.umami.is/script.js';
      script.async=true;
      script.dataset.websiteId=websiteId;
      script.dataset.domains=host;
      script.dataset.excludeSearch='true';
      script.dataset.excludeHash='true';
      script.dataset.doNotTrack='true';
      script.addEventListener('load',flush,{once:true});
      script.addEventListener('error',()=>{pending.length=0;},{once:true});
      doc.head.appendChild(script);
      return true;
    }catch(_e){
      started=false;
      pending.length=0;
      return false;
    }
  }
  function track(event){
    // Only constant names; never accept user-controlled metadata.
    if(!permitted.has(event) || !started || !active())return false;
    if(ready || typeof win.umami?.track==='function'){
      try{win.umami.track(event);return true}catch(_e){return false}
    }
    if(pending.length<24)pending.push(event);
    return true;
  }
  return {initialize,track,isActive:()=>started&&active()};
}

export const folirinaAnalytics=createAnonymousTracker();
export const trackUsage=event=>folirinaAnalytics.track(event);
