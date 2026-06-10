import {supabase} from '../config/supabaseClient.js';

const ISSUE_CATEGORIES = {
    'Water & Sanitation': ['water', 'leak', 'pipe', 'drainage', 'sewage', 'garbage', 'trash', 'waste'],
    'Electricity & Power': ['electricity', 'power', 'outage', 'blackout', 'wire', 'transformer'],
    'Infrastructure & Roads': ['road', 'pothole', 'traffic', 'signal', 'footpath', 'street'],
    'Public Safety': ['crime', 'theft', 'safety', 'harassment', 'cctv', 'emergency']
};

export function analyzeTextLogic(text) {
    if (!text) return { category: 'Other', baseSeverity: 1.0 };
    
    const cleanText = text.toLowerCase().replace(/[^a-z0-9\s]/g, '');
    let category = 'Other';
    let keywordHits = 0;

    for (const [cat, keywords] of Object.entries(ISSUE_CATEGORIES)) {
        const matches = keywords.filter(keyword => cleanText.includes(keyword));
        if (matches.length > 0) {
            category = cat;
            keywordHits += matches.length;
            break;
        }
    }

    // Normalized logic math: Base calculation on match density
    const baseSeverity = Math.min(1.0 + (keywordHits * 1.5), 10.0);
    return { category, baseSeverity };
}

export async function getIssues(req, res) {
    if (!req.dbConnected) {
        return res.status(503).json({ success: false, error: "Database service down. Admin alert flagged." });
    }

    try {
        const { data, error } = await supabase
            .from('analyzed_issues')
            .select('*')
            .order('severity_score', { ascending: false });

        if (error) throw error;
        res.json(data);
    } catch (err) {
        res.status(500).json({ success: false, error: err.message });
    }
}

export async function processManualPost(req, res) {
    if (!req.dbConnected) {
        return res.status(503).json({ success: false, error: "Database offline. Post aborted." });
    }

    try {
        const {text, user_id} = req.body;
        const { category, baseSeverity } = analyzeTextLogic(text);

        const newRecord = {
            user_id: user_id || 'anonymous',
            text: text,
            issue_category: category,
            severity_score: baseSeverity,
            upvotes: 0,
            downvotes: 0,
            created_at: new Date().toISOString()
        };

        const { data, error } = await supabase
            .from('analyzed_issues')
            .insert([newRecord])
            .select();

        if (error) throw error;
        res.status(201).json({ success: true, data: data[0] });
    } catch (err) {
        res.status(500).json({ success: false, error: err.message });
    }
}

export async function handleVote(req, res) {
    if (!req.dbConnected) {
        return res.status(503).json({ success: false, error: "Database offline. Vote unrecorded." });
    }

    try {
        const { id } = req.params;
        const { voteType } = req.body; // 'upvote' or 'downvote'
        
        if (voteType !== 'upvote' && voteType !== 'downvote') {
            return res.status(400).json({ error: "Invalid vote structure directive." });
        }

        const fieldToIncrement = voteType === 'upvote' ? 'upvotes' : 'downvotes';
        
        // Fetch current counters
        const { data: current, error: fetchErr } = await supabase
            .from('analyzed_issues')
            .select('upvotes, downvotes, severity_score')
            .eq('id', id)
            .single();

        if (fetchErr) throw fetchErr;

        const updatedCount = current[fieldToIncrement] + 1;
        
        // Adjust final visibility weights dynamically based on updated interaction telemetry
        const netVotes = (voteType === 'upvote' ? 1 : -1);
        const dynamicSeverity = Math.max(0.0, Math.min(current.severity_score + (netVotes * 0.5), 10.0));

        const { error: updateErr } = await supabase
            .from('analyzed_issues')
            .update({ 
                [fieldToIncrement]: updatedCount,
                severity_score: dynamicSeverity
            })
            .eq('id', id);

        if (updateErr) throw updateErr;
        res.json({ success: true, new_severity: dynamicSeverity });
    } catch (err) {
        res.status(500).json({ success: false, error: err.message });
    }
}
