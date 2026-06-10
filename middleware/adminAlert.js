import {supabase} from '../config/supabaseClient.js';

export async function checkSystemHealth(req, res, next) {
    try {
        //to initiate low-overhead health check query against the public issues tabular data
        const { error } = await supabase.from('analyzed_issues').select('id').limit(1);
        
        if (error) throw error;
        req.dbConnected = true;
        }   
        catch (err) 
        {
        // System logs alert structure 
        //safe to link with email notify chains later
        console.error(`\n[ADMIN NOTIFICATION] [SYSTEM FAULT ALERT]:`);
        console.error(`Timestamp: ${new Date().toISOString()}`);
        console.error(`Status   : Database infrastructure unreachable or misconfigured.`);
        console.error(`Details  : ${err.message}\n`);
        
        req.dbConnected = false;
    }
    next();
}
