import {Router} from 'express';
import {getIssues, processManualPost, handleVote } from '../controllers/nlpController.js';
import {checkSystemHealth} from '../middleware/adminAlert.js';

const router = Router();

// applying systemic verification across service access points
router.use(checkSystemHealth);

router.get('/issues', getIssues);
router.post('/issues', processManualPost);
router.post('/issues/:id/vote', handleVote);

export default router;
