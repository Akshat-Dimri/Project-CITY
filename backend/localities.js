// localities.js
// Mirrors nlp_pipeline/localities.py — kept in sync manually since it's a
// small, mostly-static reference list. Used for the /api/localities
// endpoint (map legend / reference circle) — actual complaint pins use
// the lat/lon already stored on each complaint.

const CENTER = { name: 'NIT Hamirpur Campus', lat: 31.7115, lon: 76.5117 };

const LOCALITIES = [
  { name: 'NIT Hamirpur Campus',            lat: 31.7115, lon: 76.5117 },
  { name: 'Degree College Chowk',           lat: 31.7040, lon: 76.5170 },
  { name: 'Hamirpur Town Center',           lat: 31.6908, lon: 76.5177 },
  { name: 'Green Park Colony',              lat: 31.6985, lon: 76.5140 },
  { name: 'Patel Nagar',                    lat: 31.7005, lon: 76.5210 },
  { name: 'Vegetable Market (Sabzi Mandi)', lat: 31.6950, lon: 76.5155 },
  { name: 'District Library Area',          lat: 31.6930, lon: 76.5190 },
  { name: 'Sarahkar',                       lat: 31.7050, lon: 76.5050 },
  { name: 'Majhog Sultani',                 lat: 31.7300, lon: 76.5320 },
  { name: 'Daruhi',                         lat: 31.6870, lon: 76.5340 },
  { name: 'NH-88 Bypass Road',              lat: 31.7150, lon: 76.5220 },
];

module.exports = { CENTER, LOCALITIES };
