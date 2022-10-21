mistplayers.dashjs={name:"Dash.js player",mimes:["dash/video/mp4"],priority:MistUtil.object.keys(mistplayers).length+1,isMimeSupported:function(e){return MistUtil.array.indexOf(this.mimes,e)==-1?false:true},isBrowserSupported:function(e,t,i){if(location.protocol!=MistUtil.http.url.split(t.url).protocol){i.log("HTTP/HTTPS mismatch for this source");return false}if(location.protocol=="file:"){i.log("This source ("+e+") won't load if the page is run via file://");return false}return"MediaSource"in window},player:function(){this.onreadylist=[]},scriptsrc:function(e){return e+"/dashjs.js"}};var p=mistplayers.dashjs.player;p.prototype=new MistPlayer;p.prototype.build=function(e,t){var i=this;this.onDashLoad=function(){if(e.destroyed){return}e.log("Building DashJS player..");var r=document.createElement("video");if("Proxy"in window){var a={get:{},set:{}};e.player.api=new Proxy(r,{get:function(e,t,i){if(t in a.get){return a.get[t].apply(e,arguments)}var r=e[t];if(typeof r==="function"){return function(){return r.apply(e,arguments)}}return r},set:function(e,t,i){if(t in a.set){return a.set[t].call(e,i)}return e[t]=i}});if(e.info.type=="live"){a.get.duration=function(){var t=0;if(this.buffered.length){t=this.buffered.end(this.buffered.length-1)}var i=((new Date).getTime()-e.player.api.lastProgress.getTime())*.001;return t+i+-1*e.player.api.liveOffset+45};a.set.currentTime=function(t){var i=t-e.player.api.duration;e.log("Seeking to "+MistUtil.format.time(t)+" ("+Math.round(i*-10)/10+"s from live)");e.video.currentTime=t};MistUtil.event.addListener(r,"progress",function(){e.player.api.lastProgress=new Date});e.player.api.lastProgress=new Date;e.player.api.liveOffset=0}}else{i.api=r}if(e.options.autoplay){r.setAttribute("autoplay","")}if(e.options.loop&&e.info.type!="live"){r.setAttribute("loop","")}if(e.options.poster){r.setAttribute("poster",e.options.poster)}if(e.options.muted){r.muted=true}if(e.options.controls=="stock"){r.setAttribute("controls","")}var s=dashjs.MediaPlayer().create();s.initialize(r,e.source.url,e.options.autoplay);i.dash=s;var o=["METRIC_ADDED","METRIC_UPDATED","METRIC_CHANGED","METRICS_CHANGED","FRAGMENT_LOADING_STARTED","FRAGMENT_LOADING_COMPLETED","LOG","PLAYBACK_TIME_UPDATED","PLAYBACK_PROGRESS"];for(var n in dashjs.MediaPlayer.events){if(o.indexOf(n)<0){i.dash.on(dashjs.MediaPlayer.events[n],function(t){e.log("Player event fired: "+t.type)})}}e.player.setSize=function(e){this.api.style.width=e.width+"px";this.api.style.height=e.height+"px"};e.player.api.setSource=function(t){e.player.dash.attachSource(t)};var l=false;i.dash.on("allTextTracksAdded",function(){l=true});e.player.api.setSubtitle=function(t){if(!l){var r=function(){e.player.api.setSubtitle(t);i.dash.off("allTextTracksAdded",r)};i.dash.on("allTextTracksAdded",r);return}if(!t){i.dash.enableText(false);return}var a=i.dash.getTracksFor("text");for(var s in a){var o="idx"in t?t.idx:t.trackid;if(a[s].id==o){i.dash.setTextTrack(s);if(!i.dash.isTextEnabled()){i.dash.enableText()}return true}}return false};MistUtil.event.addListener(r,"progress",function(t){if(e.container.getAttribute("data-loading")=="stalled"){e.container.removeAttribute("data-loading")}});i.api.unload=function(){i.dash.reset()};e.log("Built html");t(r)};if("dashjs"in window){this.onDashLoad()}else{var r=MistUtil.scripts.insert(e.urlappend(mistplayers.dashjs.scriptsrc(e.options.host)),{onerror:function(t){var i="Failed to load dashjs.js";if(t.message){i+=": "+t.message}e.showError(i)},onload:i.onDashLoad},e)}};