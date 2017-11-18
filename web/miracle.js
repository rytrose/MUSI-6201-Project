/*
    OSC Communication and Handlers
*/
var port = new osc.WebSocketPort({
    url: "ws://" + window.location.hostname + ":8081"
});

port.on("message", function (oscMessage) {
    // Configure handlers here
    
});

port.open();

chart = new Highcharts.chart('graph', {
    chart: {
        type: 'scatter'
    },
    title: {
        text: 'Real Time Mood'
    },
    plotOptions: {
	    scatter: {
	        allowPointSelect: false,
	        dataLabels: { enabled: false }
	    }
    },
    xAxis: {
        title: {
            enabled: true,
            text: 'Valence'
        },
        min: -1,
        max: 1,
        gridLineWidth: 1
    },
    yAxis: {
        title: {
        	enabled: true,
            text: 'Arousal'
        },
        min: -1,
        max: 1
    },
    tooltip: {
    	enabled: false
    },
    series: [{
        name: 'Mood',
        color: '#f00',
        states: {
    		hover: {
    			enabled: false
    		}
    	},
        data: [[0,0]]
    }]
});

var setDot = function(x, y, confidence) {
	// chart.plotOptions.update({
	// 	scatter: {
	// 		marker: {
	// 			radius: 4 + (4 * confidence)
	// 		}
	// 	}
	// }, true);
	chart.series[0].update({
	    data: [[x, y]],
	}, true);
	chart.series[0].data[0].update({
		marker: {
			radius: 40 - (36 * confidence)
		},
		color: hslToHex(confidence * 100, 100, 50)
	});
}

var hslToHex = function(h, s, l) {
  h /= 360;
  s /= 100;
  l /= 100;
  let r, g, b;
  if (s === 0) {
    r = g = b = l; // achromatic
  } else {
    const hue2rgb = (p, q, t) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1 / 3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1 / 3);
  }
  const toHex = x => {
    const hex = Math.round(x * 255).toString(16);
    return hex.length === 1 ? '0' + hex : hex;
  };
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}