// routes/userSkillRoutes.js
const express = require('express');
const router = express.Router();
const {
  getAllUserSkills,
  getUserSkillById,
  createUserSkill,
  updateUserSkill,
  deleteUserSkill,
  createLearningPath,
  getQuestionsAndResourcesByPathId,
  getQuestionsBySkillId,
  insertQuestionsWithLearningPath,
  updateQuestionsWithId,
  insertResources,
  getUserByPathId,
  saveUserSkillLogs,
  getResourceBySkillIdAndType,
  changeResourceAndQuestion,
  getUserKnowledgeProfile,
  changeLearningPath,
  confirmPathChange, 
  updateUserKnowledgeProfile,
  checkKnowledgePoint,
  get_temp_questions,
  checkAnswer,
  generateFinalExam
} = require('../controllers/userSkillController');

router.get('/user_skills', getAllUserSkills);
router.get('/user_skills/:id', getUserSkillById);
router.post('/user_skills', createUserSkill);
router.put('/user_skills/:id', updateUserSkill);
router.delete('/user_skills/:id', deleteUserSkill);
router.post('/learning-paths', createLearningPath);
router.post('/insertQuestions', insertQuestionsWithLearningPath);
router.post('/updateQuestions', updateQuestionsWithId);
router.post('/insertResources', insertResources);
router.get('/learning-paths/:path_id/questions_and_resources', getQuestionsAndResourcesByPathId);
router.get("/get-questions-by_skill_id/:skill_id", getQuestionsBySkillId)
router.get("/get-user-id/:path_id", getUserByPathId)
router.get('/user-knowledge-profile/:user_id', getUserKnowledgeProfile);
router.get('/update-user-knowledge-profile/:user_id', updateUserKnowledgeProfile);
router.post("/learning-paths/:path_id/submit", saveUserSkillLogs)
router.post("/get-resources-by-skill-id", getResourceBySkillIdAndType)
router.post("/learning-paths/:path_id/change-resource-and-question", changeResourceAndQuestion)
router.post("/learning-paths/:path_id/change-learning-path", changeLearningPath)
router.post("/learning-paths/:path_id/confirm-path-change", confirmPathChange)
router.post("/detection/submit", checkAnswer)
router.get("/checkKnowledge", checkKnowledgePoint)
router.get("/get_temp_questions",get_temp_questions)
router.get("/final-exam/:path_id",generateFinalExam)
module.exports = router;